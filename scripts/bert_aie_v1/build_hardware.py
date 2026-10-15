"""Finite offline whole-BERT build steps. Does not access/program physical devices."""
from pathlib import Path
import argparse, contextlib, datetime, hashlib, json, os, signal, subprocess, sys, time, shutil, struct

ROOT=Path(__file__).resolve().parents[2]
TOOLS=Path('/home/jp19126/Xilinx/2025.2/Vitis')
PLATFORM=TOOLS/'base_platforms/xilinx_vek280_base_202520_1/xilinx_vek280_base_202520_1.xpfm'
SOURCE=ROOT/'hardware/bert_aie_v1'
BUILD=ROOT/'build/bert_aie_v1'

def memory_available():
    linux=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    ps='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    value=subprocess.check_output([ps,'-NoProfile','-NonInteractive','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],timeout=15,text=True)
    return {'windows_available_bytes':int(value.strip())*1024,'wsl_available_bytes':linux}

def descendant_snapshot(root_pid):
    rows=[tuple(map(int,row.split())) for row in subprocess.check_output(
        ['ps','-eo','pid=,ppid=,rss='],text=True).splitlines()]
    owned={root_pid}
    while True:
        expanded=owned|{pid for pid,ppid,rss in rows if ppid in owned}
        if expanded==owned:break
        owned=expanded
    return {pid:rss*1024 for pid,ppid,rss in rows if pid in owned}

def run(cmd, cwd, seconds, label):
    log=cwd/(label+'.log'); record=cwd/(label+'.json')
    if record.exists():raise RuntimeError('Existing stage preserved: '+str(record))
    start=time.monotonic()
    data={'command':list(map(str,cmd)),'cwd':str(cwd),'timeout_seconds':seconds,
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'build_driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'input_artifact_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in cmd if str(p).endswith(('.xo','.a','.xpfm','.cfg')) and Path(p).is_file()},
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir() if p.is_file()},
          'boundary':'offline tool execution, no physical device access'}
    vendor=label.startswith(('aie_','aiesim_','hls_','link_'))
    vendor_only=label.startswith('aie_') and '--aie-vendor-only' in sys.argv
    data['resource_policy']={
        'aie_vendor_only':vendor_only,
        'compatibility_assumption':('Coordinator-authorized CPU/AIE overlap with GPU training; single ELF job; '
            'launch requires Windows4GiB/WSL8GiB and runtime aborts below2GiB/4GiB.' if vendor_only
            else 'AIE and link take vendor then GPU locks; other vendor stages take vendor lock. Link requires Windows8GiB/WSL8GiB at launch; runtime floors remain2GiB/4GiB.'),
        'sampled_descendant_rss_boundary':'Sum RSS for owned process and descendants across process groups; shared pages can be counted multiple times; detached/reparented children may be excluded; not physical/PSS peak.'}
    if vendor:
        data['preflight_memory']=memory_available()
        windows_launch_gib=8 if label.startswith('link_') else 4
        data['resource_policy']['windows_launch_gib']=windows_launch_gib
        if data['preflight_memory']['windows_available_bytes']<windows_launch_gib*2**30 or data['preflight_memory']['wsl_available_bytes']<8*2**30:
            data['status']='blocked_memory_preflight';record.write_text(json.dumps(data,indent=2)+'\n');raise RuntimeError(f'Vendor memory preflight below{windows_launch_gib}GiB Windows/8GiB WSL')
    record.write_text(json.dumps(data,indent=2)+'\n')
    with log.open('w') as f:
        proc=subprocess.Popen(list(map(str,cmd)),cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        data['owned_root_pid']=proc.pid;data['status']='running'
        record.write_text(json.dumps(data,indent=2)+'\n')
        try:
            if vendor:
                data['minimum_available_memory']=dict(data['preflight_memory'])
                while proc.poll() is None:
                    if time.monotonic()-start>=seconds:raise subprocess.TimeoutExpired(cmd,seconds)
                    try:sample=memory_available()
                    except Exception as error:
                        data['memory_monitor_error']=repr(error);raise subprocess.TimeoutExpired(cmd,seconds)
                    if sample['windows_available_bytes']<data['minimum_available_memory']['windows_available_bytes']:
                        data['processes_at_minimum_windows_memory']=subprocess.check_output(
                            ['ps','-eo','pid,ppid,rss,comm','--sort=-rss'],text=True).splitlines()[:12]
                    for k,v in sample.items():data['minimum_available_memory'][k]=min(data['minimum_available_memory'][k],v)
                    owned_rss=sum(descendant_snapshot(proc.pid).values())
                    data['maximum_sampled_descendant_rss_bytes']=max(data.get('maximum_sampled_descendant_rss_bytes',0),owned_rss)
                    if sample['windows_available_bytes']<2*2**30 or sample['wsl_available_bytes']<4*2**30:
                        data['resource_floor_abort']=sample;raise subprocess.TimeoutExpired(cmd,seconds)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
                data['returncode']=proc.returncode
            else:data['returncode']=proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            # Vendor launchers may create new process groups. Stop only the
            # captured owned descendant tree, including those nested groups.
            owned_pids=set(descendant_snapshot(proc.pid))
            data['terminated_owned_pids']=sorted(owned_pids)
            for pid in owned_pids:
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                for pid in owned_pids:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                proc.wait()
            data['timeout']='resource_floor_abort' not in data and 'memory_monitor_error' not in data;data['returncode']=proc.returncode
    data['elapsed_seconds']=time.monotonic()-start
    aborted=bool(data.get('timeout') or 'resource_floor_abort' in data or 'memory_monitor_error' in data)
    data['status']='aborted' if aborted else ('failed' if data['returncode'] else 'command_completed')
    record.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
    if aborted:raise SystemExit(124 if data.get('timeout') else 75)
    if data['returncode']:raise SystemExit(data['returncode'])

def fixtures(dest):
    dest.mkdir(exist_ok=True)
    real=ROOT/'results/bert_aie_v1/numerics/real_tiles'
    integer=json.loads((real/'int8.json').read_text())
    # First actual FIT-derived tile, then signed INT8 extremal values.
    IAs=[integer['A'],[[127 if (r+k)%2 else -127 for k in range(256)] for r in range(32)]]
    IBs=[integer['B'],[[-127 if (k+j)%3 else 127 for j in range(32)] for k in range(256)]]
    ias=[];ibs=[];iexpect=[]
    for A,B in zip(IAs,IBs):
        ias.extend(A[rb+r][k+t] for rb in range(0,32,4) for k in range(0,256,16) for r in range(4) for t in range(16))
        ibs.extend(B[k][cb+j] for cb in range(0,32,8) for k in range(256) for j in range(8))
        iexpect.extend(sum(A[rb+r][k]*B[k][cb+j] for k in range(256)) for rb in range(0,32,4) for cb in range(0,32,8) for r in range(4) for j in range(8))
    fas=[];fbs=[];fexpect=[];bounds=[]
    for name in ('qk32','pv32'):
        tile=json.loads((real/(name+'.json')).read_text());A=tile['A'];B=tile['B']
        fas.extend(A[rb+r][k+t] for rb in range(0,32,4) for k in range(0,64,8) for r in range(4) for t in range(8))
        fbs.extend(B[k][cb+j] for cb in range(0,32,8) for k in range(64) for j in range(8))
        fexpect.extend(tile['expected_float64'][rb+r][cb+j] for rb in range(0,32,4) for cb in range(0,32,8) for r in range(4) for j in range(8))
        bounds.extend(tile['fp32_absolute_error_bound'][rb+r][cb+j] for rb in range(0,32,4) for cb in range(0,32,8) for r in range(4) for j in range(8))
    for name,vals,width in [('i8_a',ias,16),('i8_b',ibs,16),('bf_a',fas,8),('bf_b',fbs,8)]:
        (dest/(name+'.txt')).write_text('\n'.join(' '.join(format(x,'.9g') for x in vals[i:i+width]) for i in range(0,len(vals),width))+'\n')
    (dest/'expected.json').write_text(json.dumps({'int8':iexpect,'bf16':fexpect,'bf16_bounds':bounds,
        'source_manifest':json.loads((real/'manifest.json').read_text())},indent=2)+'\n')

def check_sim(dst):
    import math
    expected=json.loads((dst/'data/expected.json').read_text());out={}
    for key,file in [('int8','i8_c.txt'),('bf16','bf_c.txt')]:
        values=[]
        for line in (dst/'aiesimulator_output/data'/file).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith(('T','#')):continue
            values.extend(float(x) for x in line.split())
        gold=expected[key]
        if len(values)!=len(gold):raise RuntimeError(f'{key}: output count {len(values)} != {len(gold)}')
        errors=[abs(x-y) for x,y in zip(values,gold)]
        bounds=[0]*len(gold) if key=='int8' else expected['bf16_bounds']
        passed=all(math.isfinite(x) and error<=bound for x,error,bound in zip(values,errors,bounds))
        out[key]={'count':len(values),'passed':passed,'max_absolute_error':max(errors),
                  'max_bound_ratio':max((e/b if b else (0 if e==0 else float('inf'))) for e,b in zip(errors,bounds))}
    out['boundary']='actual AIE simulator real FIT tiles plus INT8 boundary tile; not full-model hardware simulation'
    (dst/'aiesim_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
    if not all(out[k]['passed'] for k in ('int8','bf16')):raise RuntimeError('AIE arithmetic check failed')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['cpu','cpu-hls','wire','aie','aiesim','check-sim','hls','link'])
    ap.add_argument('--mapping',choices=['resident','external'],default='resident')
    ap.add_argument('--attempt',type=int,default=0)
    ap.add_argument('--timeout',type=int,default=900)
    ap.add_argument('--aie-vendor-only',action='store_true',help='Coordinator-authorized AIE/GPU overlap under unchanged memory guards')
    args=ap.parse_args()
    if args.aie_vendor_only and args.stage!='aie':ap.error('--aie-vendor-only applies only to AIE compilation')
    if os.name=='nt':raise RuntimeError('Use Linux Python after sourcing installed Vitis settings64.sh')
    if not 1<=args.timeout<=7200:raise ValueError('Finite timeout in [1,7200] required')
    dst=BUILD/(args.mapping if args.stage in ('hls','link') else 'common');dst.mkdir(parents=True,exist_ok=True)
    label=f'{args.stage}_attempt{args.attempt}'
    if (dst/(label+'.json')).exists():raise RuntimeError('Existing stage preserved: '+label)
    if args.stage=='cpu':
        run(['g++','-O3','-std=c++17','-ffp-contract=off',SOURCE/'functional.cpp','-o',dst/'bert_functional'],dst,args.timeout,label)
    elif args.stage=='cpu-hls':
        sim=TOOLS.parent/'lnx64/lib/csim';fpo=TOOLS/'lnx64/tools/fpo_v7_1'
        run(['g++','-O3','-std=c++17','-ffp-contract=off','-DBERT_HLS','-I'+str(TOOLS/'include'),SOURCE/'functional.cpp',
             '-L'+str(sim),'-lhlsmc++-GCC46','-L'+str(fpo),'-lIp_floating_point_v7_1_bitacc_cmodel',
             '-Wl,-rpath,'+str(sim),'-Wl,-rpath,'+str(fpo),'-Wl,-rpath-link,'+str(fpo),'-o',dst/'bert_functional_hls'],dst,args.timeout,label)
    elif args.stage=='wire':
        run(['g++','-O2','-std=c++17','-ffp-contract=off','-I'+str(TOOLS/'include'),SOURCE/'wire_test.cpp','-o',dst/'wire_test'],dst,args.timeout,label)
        run([dst/'wire_test'],dst,60,label+'_run')
    elif args.stage=='aie':
        if (dst/'Work').exists():
            saved=dst/('preserved_before_aie_attempt'+str(args.attempt));saved.mkdir()
            for name in ('Work','graph.cpp','tiles.cpp','tiles.hpp','data','libadf.a'):
                if (dst/name).exists():(dst/name).rename(saved/name)
        for name in ('tiles.hpp','tiles.cpp','graph.cpp'):shutil.copyfile(SOURCE/name,dst/name)
        fixtures(dst/'data')
        run([TOOLS/'aietools/bin/aiecompiler','--target=hw','--platform='+str(PLATFORM),'--workdir=Work','--Xelfgen=-j1','--include='+str(dst),dst/'graph.cpp'],dst,args.timeout,label)
    elif args.stage=='aiesim':
        run([TOOLS/'aietools/bin/aiesimulator','--pkg-dir=Work','--input-dir=.','--enable-memory-check','--simulation-cycle-timeout=200000'],dst,args.timeout,label)
    elif args.stage=='check-sim':check_sim(dst)
    elif args.stage=='hls':
        if (dst/'hls').exists():
            saved=dst/('preserved_before_hls_attempt'+str(args.attempt));saved.mkdir()
            for name in ('hls','hls.cfg','bert_pl.xo','hls_costs.json'):
                if (dst/name).exists():(dst/name).rename(saved/name)
        cfg=dst/'hls.cfg'
        cfg.write_text('part=xcve2802-vsvh1760-2MP-e-S\n[hls]\nflow_target=vitis\nsyn.top=bert_pl\nsyn.file='+str(SOURCE/'pl.cpp')+'\nsyn.cflags=-std=c++14 -ffp-contract=off -DBERT_RESIDENT='+str(int(args.mapping=='resident'))+'\nclock=4ns\npackage.output.syn=true\npackage.output.format=xo\npackage.output.file='+str(dst/'bert_pl.xo')+'\n')
        # Only the pointwise normalization output loop: distinct c iterations
        # access distinct x elements; gamma/beta are separate immutable BOs.
        # Mean/variance reduction dependence and source arithmetic stay intact.
        with cfg.open('a') as f:f.write('syn.directive.dependence=variable=x type=inter dependent=false bert_aie::norm/VITIS_LOOP_49_4\n')
        run([TOOLS/'bin/v++','--compile','--mode','hls','--config',cfg,'--work_dir',dst/'hls'],dst,args.timeout,label)
    else:
        if (dst/'link_tmp').exists():
            saved=dst/('preserved_before_link_attempt'+str(args.attempt));saved.mkdir()
            for name in ('link_tmp','bert.xsa'):
                if (dst/name).exists():(dst/name).rename(saved/name)
        run([TOOLS/'bin/v++','--link','--target','hw','--platform',PLATFORM,'--config',SOURCE/'system.cfg',
             '--save-temps','--temp_dir',dst/'link_tmp','--output',dst/'bert.xsa',dst/'bert_pl.xo',BUILD/'common/libadf.a'],dst,args.timeout,label)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1] in ('aie','aiesim','hls','link'):
        from common import lock
        with contextlib.ExitStack() as resources:
            resources.enter_context(lock('vendor'))
            if sys.argv[1]=='link' or (sys.argv[1]=='aie' and '--aie-vendor-only' not in sys.argv):resources.enter_context(lock('gpu0',timeout=900))
            main()
    else:main()
