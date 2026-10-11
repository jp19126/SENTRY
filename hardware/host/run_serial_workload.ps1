param(
    [Parameter(Mandatory=$true)][string]$Workload,
    [string]$Port,
    [string]$Output,
    [string]$RuntimeSettings,
    [switch]$AllowHardware
)
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).ProviderPath
$config=Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'configs/project.json') | ConvertFrom-Json
# This continuation always leaves the shared physical-activity gate false.
if($AllowHardware -and $config.execution_control.board_free_continuation.physical_board_activity_authorized -ne $true){
    throw 'Physical board activity is not authorized in the shared configuration.'
}
if($AllowHardware){
    if(-not $RuntimeSettings){throw 'Approved actual runtime settings are required'}
    $runtime=Get-Content -Raw -LiteralPath $RuntimeSettings | ConvertFrom-Json
    if($runtime.approved -ne $true -or -not $runtime.unit_id -or -not $runtime.board_revision -or -not $runtime.sd_media_identity -or $runtime.sd_slot_reserved -ne $true -or $runtime.arena_allocation_confirmed -ne $true -or -not $runtime.profile_id -or $runtime.timer_frequency_verified -ne $true -or $runtime.uart_port -ne $Port){throw 'Actual unit/media/arena/profile/timer/port settings unresolved'}
    $bootPackage=Get-Content -Raw -LiteralPath $runtime.boot_package | ConvertFrom-Json
    $closurePath=$bootPackage.integrated_closure
    if($closurePath -and -not [IO.Path]::IsPathRooted($closurePath)){$closurePath=Join-Path $taskRoot $closurePath}
    if($bootPackage.integrated_eligible -ne $true -or -not $closurePath -or -not (Test-Path -LiteralPath $closurePath)){throw 'Actual routed timing/resource/bandwidth eligibility unresolved'}
    if($bootPackage.passed -ne $true -or $bootPackage.boot_packaged -ne $true -or $runtime.arena_base -ne $bootPackage.deployment.arena_base -or $runtime.profile_id -ne $bootPackage.profile_id -or $runtime.timer_frequency_hz -ne $bootPackage.timer_frequency_hz){throw 'Approved settings differ from compiled boot package'}
}
function Read-Exact($stream,[byte[]]$buffer,[int]$count){
    $offset=0
    while($offset -lt $count){
        $n=$stream.Read($buffer,$offset,$count-$offset)
        if($n -eq 0){throw 'Truncated packet/response'}
        $offset+=$n
    }
}
$inputStream=[System.IO.File]::OpenRead((Resolve-Path -LiteralPath $Workload).ProviderPath)
$serial=$null;$records=[System.Collections.Generic.List[object]]::new()
$active=$false;$expectedIndex=0;$documents=0;$packets=0;$fixedModel=$null;$fixedThreshold=$null;$fixedLength=$null
try{
    if($AllowHardware){
        if(-not $Port -or -not $Output){throw 'Hardware execution requires explicit Port and a fresh Output path'}
        if(Test-Path -LiteralPath $Output){throw 'Preserve prior measurements; choose a fresh Output'}
        $serial=[System.IO.Ports.SerialPort]::new($Port,115200,[System.IO.Ports.Parity]::None,8,[System.IO.Ports.StopBits]::One)
        $serial.ReadTimeout=120000;$serial.WriteTimeout=120000;$serial.Handshake=[System.IO.Ports.Handshake]::None
        $serial.Open()
        $serial.Write([byte[]]@(0xa5),0,1)
        $ready=[byte[]]::new(64);Read-Exact $serial $ready 64
        if([Text.Encoding]::ASCII.GetString($ready,0,8) -ne "SENTRYR1" -or [BitConverter]::ToUInt32($ready,8) -ne 1 -or [BitConverter]::ToUInt32($ready,28) -ne $runtime.timer_frequency_hz){throw "Supervisor initialization handshake failed"}
    }
    while($inputStream.Position -lt $inputStream.Length){
        $header=[byte[]]::new(64);Read-Exact $inputStream $header 64
        $length=[BitConverter]::ToUInt32($header,12);$valid=[BitConverter]::ToUInt32($header,16)
        $index=[BitConverter]::ToUInt32($header,20);$doc=[BitConverter]::ToUInt64($header,24)
        $sample=[BitConverter]::ToUInt64($header,32);$model=[BitConverter]::ToUInt64($header,40)
        $flags=[BitConverter]::ToUInt32($header,48);$bytes=[BitConverter]::ToUInt32($header,52)
        $threshold=[BitConverter]::ToUInt32($header,56)
        if([Text.Encoding]::ASCII.GetString($header,0,8) -ne 'SENTRY01' -or [BitConverter]::ToUInt32($header,8) -ne 1 -or
           $length -notin 128,256,512 -or $valid -lt 1 -or $valid -gt $length -or $flags -gt 1 -or
           $bytes -ne 9*$length -or $threshold -gt 0x3f800000 -or [BitConverter]::ToUInt32($header,60) -ne 0){throw 'Invalid packet header'}
        if($null -eq $fixedModel){$fixedModel=$model;$fixedThreshold=$threshold;$fixedLength=$length;if($AllowHardware -and ([BitConverter]::ToUInt32($ready,12) -ne $length -or [BitConverter]::ToUInt64($ready,16) -ne $model -or [BitConverter]::ToUInt32($ready,24) -ne $threshold)){throw "Ready model/length/threshold mismatch"}}
        if($model -ne $fixedModel -or $threshold -ne $fixedThreshold -or $length -ne $fixedLength){throw 'Frozen model/threshold mismatch'}
        if(-not $active){if($index -ne 0){throw 'New document must start at window0'};$document=$doc;$expectedIndex=0;$documents++;$maximum=[uint32]0}
        if($doc -ne $document -or $index -ne $expectedIndex){throw 'Document window sequence mismatch'}
        $payload=[byte[]]::new($bytes);Read-Exact $inputStream $payload $bytes
        for($i=0;$i -lt $length;$i++){
            $wanted=0;if($i -lt $valid){$wanted=1}
            if([BitConverter]::ToUInt32($payload,4*$i) -ge 30522 -or [BitConverter]::ToUInt32($payload,4*$length+4*$i) -gt 1 -or $payload[8*$length+$i] -ne $wanted){throw 'Invalid ID/type/mask'}
        }
        if($AllowHardware){
            $timer=[Diagnostics.Stopwatch]::StartNew()
            $serial.Write($header,0,64);$serial.Write($payload,0,$payload.Length)
            $response=[byte[]]::new(64);Read-Exact $serial $response 64;$timer.Stop()
            $status=[BitConverter]::ToUInt32($response,12);$risk=[BitConverter]::ToUInt32($response,44)
            $maxBits=[BitConverter]::ToUInt32($response,48);$resultFlags=[BitConverter]::ToUInt32($response,52)
            if([Text.Encoding]::ASCII.GetString($response,0,8) -ne 'SENTRES1' -or [BitConverter]::ToUInt32($response,8) -ne 1 -or
               [BitConverter]::ToUInt64($response,16) -ne $sample -or [BitConverter]::ToUInt64($response,24) -ne $doc -or
               [BitConverter]::ToUInt64($response,32) -ne $model -or [BitConverter]::ToUInt32($response,40) -ne $index -or
               ($resultFlags -band 1) -ne $flags -or $status -ne 0 -or $risk -gt 0x3f800000 -or $resultFlags -gt 3){throw 'Invalid/error/identity-mismatched response'}
            if($risk -gt $maximum){$maximum=$risk}
            $reject=($flags -eq 1 -and $maximum -gt $threshold)
            if($maxBits -ne $maximum -or (($resultFlags -band 2) -ne 0) -ne $reject){throw 'Document maximum/strict threshold mismatch'}
            $records.Add([pscustomobject]@{sample=$sample;document=$doc;window=$index;length=$length;valid=$valid;last=$flags;model=$model;risk_bits=$risk;maximum_bits=$maxBits;reject=$reject;warm_packet_to_decision_ms=$timer.Elapsed.TotalMilliseconds;kernel_submit_to_completion_ns=[BitConverter]::ToUInt64($response,56)})
        }
        $packets++;$expectedIndex++;$active=($flags -eq 0)
    }
    if($active -or $packets -eq 0){throw 'Incomplete document or empty workload'}
    $result=[ordered]@{passed=$true;packets=$packets;documents=$documents;physical_board_activity=[bool]$AllowHardware;model=$fixedModel;threshold_bits=$fixedThreshold;approved_runtime_settings=$runtime;records=$records}
    $json=$result | ConvertTo-Json -Depth 8
    if($AllowHardware){[IO.File]::WriteAllText([IO.Path]::GetFullPath($Output),$json)}
    $json
}finally{
    $inputStream.Dispose()
    if($null -ne $serial){if($serial.IsOpen){$serial.Close()};$serial.Dispose()}
}

