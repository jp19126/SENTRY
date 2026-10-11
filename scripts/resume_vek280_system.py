"""Resume the same frozen system after a resource stop, serializing unfinished runs.
Completed IP DCPs are retained. No source regeneration, HLS, board connection or programming.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,shutil,signal,subprocess,time,ctypes
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--system-dir',type=Path,required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--limit-seconds',type=int,default=7200);a=ap.parse_args()
 system=a.system_dir if a.system_dir.is_absolute() else ROOT/a.system_dir;system=system.resolve()
 if not system.is_relative_to((ROOT/'build/vek280_system').resolve()):raise ValueError('Resume must stay in the intended system build tree')
 original=json.loads((system/'run.json').read_text());project=system/'project/sentry_vek280.xpr';runs=system/'project/sentry_vek280.runs'
 if original['phase']!='implement' or not project.is_file() or not 1<=a.limit_seconds<=7200:raise ValueError('Actual bounded existing implementation required')
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out=system/('resume_'+stamp);out.mkdir()
 completed=[];unfinished=[]
 for run in runs.iterdir():
  if not run.is_dir() or run.name.startswith('.'):continue
  dcps=list(run.glob('*.dcp'))
  if (run/'.vivado.end.rst').is_file() and dcps:completed.append(dict(run=run.name,dcps=[str(p) for p in dcps]))
  else:
   unfinished.append(run.name)
   # Reset regenerates run scripts/logs, so preserve the entire interrupted run.
   shutil.copytree(run,out/'interrupted_runs'/run.name)
 hook=out/'limit_threads.tcl';hook.write_text('set_param general.maxThreads 2\nset_param synth.maxThreads 1\nputs "SENTRY_ACTUAL_MAX_THREADS [get_param general.maxThreads]"\nputs "SENTRY_ACTUAL_SYNTH_MAX_THREADS [get_param synth.maxThreads]"\nif {[get_param synth.maxThreads] != 1} {error "Inner synthesis worker cap not applied"}\n')
 def lit(p):return '{'+str(p).replace('\\','/')+'}'
 lines=['set_param general.maxThreads 2','set_param synth.maxThreads 1','open_project '+lit(project),'set fh [open '+lit(out/'run_status_before.tsv')+' w]',
  'foreach r [get_runs] {puts $fh "$r\t[get_property STATUS $r]\t[get_property PROGRESS $r]"}','close $fh',
  'set completed [list '+' '.join(lit(r['run']) for r in completed)+']',
  'foreach r [get_runs -filter {IS_SYNTHESIS == 1}] {',
  ' if {$r eq "synth_1" || [lsearch -exact $completed $r]>=0} {continue}',
  ' reset_run $r',
  ' set_property STEPS.SYNTH_DESIGN.TCL.PRE '+lit(hook)+' $r',
  ' launch_runs $r -jobs 1',' wait_on_run $r',
  ' if {[get_property PROGRESS $r] ne "100%"} {error "Serial IP synthesis failed for $r"}',
  '}',
  'if {[lsearch -exact $completed synth_1]<0} { reset_run synth_1; set_property STEPS.SYNTH_DESIGN.TCL.PRE '+lit(hook)+' [get_runs synth_1]; launch_runs synth_1 -jobs 1; wait_on_run synth_1 }',
  'if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Integrated synthesis failed"}',
  'set ir [get_runs impl_1]',
  'foreach property [list_property $ir] {if {[string match "STEPS.*.TCL.PRE" $property]} {set_property $property '+lit(hook)+' $ir}}',
  'launch_runs impl_1 -to_step write_device_image -jobs 1','wait_on_run impl_1',
  'if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Integrated implementation failed"}',
  'open_run impl_1','report_utilization -file '+lit(system/'integrated_utilization.rpt'),
  'report_timing_summary -file '+lit(system/'integrated_timing.rpt'),'report_drc -file '+lit(system/'integrated_drc.rpt'),
  'write_hw_platform -fixed -include_bit -force -file '+lit(system/'sentry_vek280.xsa'),
  'set fh [open '+lit(out/'run_status_after.tsv')+' w]','foreach r [get_runs] {puts $fh "$r\t[get_property STATUS $r]\t[get_property PROGRESS $r]"}','close $fh','close_project','exit']
 tcl=out/'resume.tcl';tcl.write_text('\n'.join(lines)+'\n')
 config=json.loads((ROOT/'configs/project.json').read_text(encoding='utf-8-sig'))
 command=['taskset','-c','0-5',config['hardware']['hls']['vivado_executable'],'-mode','batch','-source',str(tcl),'-log',str(out/'vendor.log'),'-journal',str(out/'vendor.jou')]
 shutil.copyfile(Path(__file__),out/'executed_driver.py')
 record=dict(phase='implement',length=original['length'],lanes=original['lanes'],system_directory=str(system),source_snapshot=original['source_snapshot'],plan=original['plan'],recorded_utc=datetime.now(timezone.utc).isoformat(),state='prepared',passed=False,cause='Actual concurrent synthesis exhausted Windows/WSL available memory; resume unfinished frozen runs serially',original_run=str(system/'run.json'),completed_runs_reused=completed,interrupted_runs_preserved=unfinished,command=command,jobs=1,max_threads=2,synthesis_workers=1,runtime_wsl_floor_kib=4*1024*1024,runtime_windows_floor_kib=2*1024*1024,cpu_affinity=list(range(6)),limit_seconds=a.limit_seconds,board_access=False,physical_measurement=False)
 def save():(out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
 save();print(json.dumps(dict(output=str(out),**record),indent=2),flush=True)
 if not a.execute:return
 if os.name=='nt':raise RuntimeError('Use WSL Linux Python')
 pause=ROOT/'results/goal2/pause.request';proc=None
 try:
  if pause.exists():raise RuntimeError('Pause marker present')
  mem={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith(('MemTotal:','MemAvailable:','SwapFree:','MemFree:','Cached:','SReclaimable:','Dirty:','Writeback:'))}
  # Host free physical memory is observed without changing Windows/WSL settings.
  host=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],text=True,capture_output=True,timeout=30)
  if host.returncode:raise RuntimeError('Cannot observe Windows headroom')
  host_free=int(host.stdout.strip());record['resource_preflight']=dict(observed_utc=datetime.now(timezone.utc).isoformat(),linux_memory_kib=mem,windows_free_kib=host_free,parent_affinity=sorted(os.sched_getaffinity(0)))
  if mem['MemAvailable']<8*1024*1024 or host_free<4*1024*1024 or not set(range(6))<=os.sched_getaffinity(0):raise RuntimeError('Resume needs8GiB WSL and4GiB Windows available headroom, CPUs0..5')
  record['state']='running';save()
  with (out/'vendor.stdout.log').open('w') as log:
   proc=subprocess.Popen(command,cwd=system,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['vendor_pid']=proc.pid;save();begin=time.monotonic();last_observe=begin
   while proc.poll() is None:
    if pause.exists() or time.monotonic()-begin>a.limit_seconds:raise RuntimeError('Pause marker present' if pause.exists() else 'Bounded resume execution limit reached')
    if 'actual_vendor_affinity' not in record and time.monotonic()-begin>.5:record['actual_vendor_affinity']=sorted(os.sched_getaffinity(proc.pid));save()
    if time.monotonic()-last_observe>5:
     current={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith(('MemAvailable:','SwapFree:','MemFree:','Cached:','SReclaimable:','Dirty:','Writeback:'))}
     observed_host=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],text=True,capture_output=True,timeout=10)
     if observed_host.returncode:raise RuntimeError('Runtime resource observation failed; preserve assigned vendor work')
     current['windows_free_kib']=int(observed_host.stdout.strip())
     process_result=subprocess.run(['ps','-eo','pid,ppid,pgid,stat,rss,comm'],capture_output=True,text=True,timeout=5,check=True)
     assigned=[]
     for line in process_result.stdout.splitlines()[1:]:
      fields=line.split()
      if len(fields)==6 and int(fields[2])==proc.pid:assigned.append(dict(pid=int(fields[0]),ppid=int(fields[1]),pgid=int(fields[2]),state=fields[3],rss_kib=int(fields[4]),command_name=fields[5]))
     observation=dict(observed_utc=datetime.now(timezone.utc).isoformat(),**current,assigned_vendor_processes=assigned)
     with (out/'resource_observations.jsonl').open('a',encoding='utf-8') as observations:observations.write(json.dumps(observation)+'\n')
     record['latest_memory_kib']={k:v for k,v in observation.items() if k!='assigned_vendor_processes'};save();last_observe=time.monotonic()
     if current['MemAvailable']<4*1024*1024 or current['windows_free_kib']<2*1024*1024:raise RuntimeError('Resource guard: runtime needs4GiB WSL and2GiB Windows available memory')
    time.sleep(.5)
  record['returncode']=proc.returncode
  if proc.returncode:raise RuntimeError('Vivado resume failed; inspect preserved logs')
  if not (system/'sentry_vek280.xsa').is_file() or not (system/'integrated_timing.rpt').is_file():raise RuntimeError('Missing actual implementation outputs')
  record.update(state='returned',passed=True)
 except BaseException as exc:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  record.update(state='stopped' if proc is not None and proc.returncode and proc.returncode<0 else 'failed',error=str(exc));raise
 finally:save()
if __name__=='__main__':main()
