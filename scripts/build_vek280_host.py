"""Bounded offline selected-XSA BSP and supervisor ELF build; no board access."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,shutil,signal,subprocess,time
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--system-dir',type=Path,required=True)
    ap.add_argument('--arena-record',type=Path,default=Path('results/goal4/internal_integration/arena_w8_qat_real_input_v1/preparation.json'))
    ap.add_argument('--model-id',type=lambda x:int(x,0),required=True)
    ap.add_argument('--sd-lba',type=int,required=True)
    ap.add_argument('--arena-base',type=lambda x:int(x,0),default=0x40000000)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--limit-seconds',type=int,default=900)
    a=ap.parse_args()
    system=a.system_dir if a.system_dir.is_absolute() else ROOT/a.system_dir
    result=json.loads((system/'run.json').read_text());plan=json.loads((system/'command_plan.json').read_text())
    if not result.get('passed'):raise ValueError('Selected system preparation must have passed')
    xsa=system/'sentry_vek280_preimplementation.xsa'
    if result['phase']=='implement':xsa=system/'sentry_vek280.xsa'
    if not xsa.is_file():raise ValueError('Selected XSA missing')
    arena_path=a.arena_record if a.arena_record.is_absolute() else ROOT/a.arena_record
    arena=json.loads(arena_path.read_text())
    if not arena.get('complete') or arena['image_bytes']!=67108864 or arena['written_buffers']['token_ids']['bytes']!=4*result['length']:raise ValueError('Frozen arena/plan length mismatch')
    if not 0<=a.model_id<2**64 or not 1<=a.sd_lba<2**32 or not 1<=a.limit_seconds<=900 or a.arena_base&4095:raise ValueError('Invalid bounded deployment choices')
    mappings=[]
    for line in (system/'address_segments.tsv').read_text().splitlines():
        space,segment,offset,size=line.split('\t')
        if offset:mappings.append((space,segment,int(offset,0),int(size,0)))
    shared=[row for row in mappings if 'DDR_LOW0' in row[1]]
    if len(shared)!=16 or any(not base<=a.arena_base or a.arena_base+67108864>base+size for _,_,base,size in shared):raise ValueError('Arena must fit every actual shared PS/PL DDR mapping')
    mailbox=[row for row in mappings if row[0]=='/cips/M_AXI_FPD' and 'mailbox' in row[1]]
    if len(mailbox)!=1:raise ValueError('Need one actual CPU mailbox mapping')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out=ROOT/'build/vek280_host'/f'l{result["length"]}'/stamp;out.mkdir(parents=True)
    sources=out/'sources';sources.mkdir()
    for name in ['main.c','sentry_supervisor.c','sentry_supervisor.h']:shutil.copyfile(ROOT/'hardware/host'/name,sources/name)
    shutil.copyfile(ROOT/'scripts/generate_vek280_bsp.py',out/'generate_bsp.py')
    offsets=plan['buffers']
    header='#include "xparameters.h"\n#include "sentry_supervisor.h"\nconst SentryDeployment deployment={\n'
    values=dict(mailbox_base=mailbox[0][2],arena_base=a.arena_base,model_id=a.model_id,mailbox_poll_limit=10000000000,tokens=result['length'],arena_bytes=67108864,arena_sd_lba=a.sd_lba,token_offset=offsets['token_ids']['offset'],type_offset=offsets['type_ids']['offset'],mask_offset=offsets['attention_mask']['offset'],threshold_bits=int(arena['threshold_bits_hex'],16),w8_mask=arena['config_w8_mask'])
    for name,value in values.items():header+=f' .{name}={value}ULL,\n'
    header+=' .uart_base=XPAR_XUARTPSV_0_BASEADDR,\n .sd_base=XPAR_XSDPS_0_BASEADDR\n};\n';(sources/'deployment.h').write_text(header)
    job=dict(output=str(out),workspace=str(out/'workspace'),xsa=str(xsa),sources=str(sources),system_run=str(system/'run.json'),arena_record=str(arena_path),selected_deployment=values,physical_allocation_confirmed=False)
    (out/'job.json').write_text(json.dumps(job,indent=2)+'\n')
    command=['/home/jp19126/Xilinx/2025.2/Vitis/bin/vitis','-s',str(out/'generate_bsp.py')]
    record=dict(state='prepared',passed=False,job=str(out/'job.json'),command=command,limit_seconds=a.limit_seconds,board_access=False)
    def save():(out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    save();print(json.dumps(dict(output=str(out),**record),indent=2),flush=True)
    if not a.execute:return
    if os.name=='nt':raise RuntimeError('Execute using WSL Linux Python')
    pause=ROOT/'results/goal2/pause.request';proc=None
    try:
        if pause.exists():raise RuntimeError('Pause marker present')
        env=os.environ.copy();env['SENTRY_HOST_JOB']=str(out/'job.json');record['state']='running';save()
        with (out/'vendor.stdout.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);began=time.monotonic()
            while proc.poll() is None:
                if pause.exists() or time.monotonic()-began>a.limit_seconds:raise RuntimeError('Pause or bounded host build limit')
                time.sleep(.5)
        if proc.returncode:raise RuntimeError('Vitis failed; inspect vendor.stdout.log')
        host=json.loads((out/'host_result.json').read_text())
        if not host.get('passed'):raise RuntimeError('Host build result missing success')
        record.update(state='returned',passed=True,host_result=host)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        record.update(state='failed',error=str(exc));raise
    finally:save()
if __name__=='__main__':main()
