"""Bounded readiness wait for one authorized stage; never steals shared leases."""
import argparse, datetime, json, os, subprocess, sys, time
from pathlib import Path
from build_hardware import ROOT, BUILD, memory_available

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['hls','link'])
    ap.add_argument('--mapping',choices=['resident','external'],required=True)
    ap.add_argument('--attempt',type=int,required=True)
    ap.add_argument('--wait-seconds',type=int,default=900)
    ap.add_argument('--timeout',type=int,default=1800)
    args=ap.parse_args()
    if not 1<=args.wait_seconds<=1800:raise ValueError('Wait ceiling1800s')
    dst=BUILD/args.mapping;dst.mkdir(parents=True,exist_ok=True)
    record=dst/f'{args.stage}_attempt{args.attempt}_readiness.json'
    if record.exists():raise RuntimeError('Preserve existing readiness record')
    locks=ROOT/'results/dual_track_v1/shared/locks'
    start=time.monotonic()
    data={'waiter_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'wait_seconds':args.wait_seconds,'child_launched':False,'status':'waiting'}
    while True:
        data['elapsed_wait_seconds']=time.monotonic()-start
        data['available_memory']=memory_available()
        data['occupied_required_leases']=[n for n in ('vendor','gpu0')
            if (n=='vendor' or args.stage=='link') and (locks/(n+'.lock')).exists()]
        ready=(not data['occupied_required_leases'] and
            data['available_memory']['windows_available_bytes'] >= (8 if args.stage=='link' else 4)*2**30 and
            data['available_memory']['wsl_available_bytes'] >= 8*2**30)
        if ready:
            cmd=[sys.executable,str(Path(__file__).with_name('build_hardware.py')),args.stage,
                 '--mapping',args.mapping,'--attempt',str(args.attempt),'--timeout',str(args.timeout)]
            data['command']=cmd;data['status']='dispatching';record.write_text(json.dumps(data,indent=2)+'\n')
            # Atomic lease acquisition and a second memory preflight happen in
            # the build driver; this observation cannot confer lock ownership.
            proc=subprocess.Popen(cmd)
            data['child_launched']=True;data['child_driver_pid']=proc.pid
            data['status']='driver_running';record.write_text(json.dumps(data,indent=2)+'\n')
            result=proc.wait();data['returncode']=result;data['status']='driver_finished'
            record.write_text(json.dumps(data,indent=2)+'\n');return result
        if data['elapsed_wait_seconds']>=args.wait_seconds:
            data['status']='readiness_timeout';record.write_text(json.dumps(data,indent=2)+'\n');return 75
        record.write_text(json.dumps(data,indent=2)+'\n');time.sleep(10)

if __name__=='__main__':raise SystemExit(main())
