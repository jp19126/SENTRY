"""Coordinator-dispatched bounded child with canonical cross-track resource lock."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from common import OUT, CANONICAL, lock, write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--resource',choices=('gpu0','cpu'),required=True)
    p.add_argument('--timeout',type=int,required=True)
    p.add_argument('--lock-wait',type=int,default=0)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if Path(a.name).name!=a.name:raise ValueError('Simple job name required')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or a.timeout<=0:raise ValueError('Need positive timeout and command')
    out=OUT/'dispatch'/a.name
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise RuntimeError('Preserve prior job; choose new name')
    out.mkdir()
    start=time.monotonic()
    result={'command':command,'cwd':str(CANONICAL),'resource':a.resource,'timeout_seconds':a.timeout,'started_unix':time.time(),'pid':os.getpid()}
    try:
        with lock(a.resource,timeout=a.lock_wait):
            start=time.monotonic()
            result['acquired_unix']=time.time()
            sys.path.insert(0,str(CANONICAL/'scripts'))
            from r1_challengers import resource_check
            result['preflight']=resource_check(True)
            write(out/'run.json',result)
            with (out/'stdout.log').open('w',encoding='utf-8') as stdout, (out/'stderr.log').open('w',encoding='utf-8') as stderr:
                child=subprocess.Popen(command,cwd=CANONICAL,stdout=stdout,stderr=stderr)
                result['child_pid']=child.pid;write(out/'run.json',result)
                try:
                    last=time.monotonic()
                    while child.poll() is None:
                        if time.monotonic()-start>a.timeout:raise TimeoutError('Bounded dispatch elapsed')
                        if time.monotonic()-last>=60:
                            result.setdefault('runtime_memory',[]).append(resource_check(False));last=time.monotonic()
                            write(out/'run.json',result)
                        time.sleep(.5)
                    result['exit_code']=child.returncode
                except BaseException:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise
            result['complete']=result['exit_code']==0
    except BaseException as error:
        result['complete']=False;result['error']=repr(error)
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        write(out/'run.json',result)
        print(result,flush=True)
    raise SystemExit(result['exit_code'])

if __name__=='__main__':main()
