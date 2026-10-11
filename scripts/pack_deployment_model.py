"""Pack one frozen arena into the supervisor's read-only SD slot; optional offline BOOT.BIN.
No media writes, programming, device discovery or board access.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,shutil,struct,subprocess,os,math
ROOT=Path(__file__).resolve().parents[1]
def resolve(p):
 p=Path(p);return p if p.is_absolute() else ROOT/p
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--arena-record',required=True);ap.add_argument('--deployment-job',required=True)
 ap.add_argument('--output',required=True);ap.add_argument('--implementation-record',help='Explicit successful same-project resume run, when the initial run was resource-stopped');ap.add_argument('--system-dir');ap.add_argument('--host-link-dir');ap.add_argument('--integrated-closure',help='Inspected actual routed system timing/resource/bandwidth eligibility record')
 ap.add_argument('--execute',action='store_true');ap.add_argument('--bootgen',action='store_true')
 a=ap.parse_args();arena_record=resolve(a.arena_record);prep=load(arena_record);job=load(resolve(a.deployment_job));d=job['selected_deployment'];out=resolve(a.output)
 if not prep.get('complete') or prep['image_bytes']!=67108864:raise ValueError('Complete frozen64MiB arena required')
 image=arena_record.parent/prep['image']
 if image.stat().st_size!=67108864:raise ValueError('Arena file size mismatch')
 if d['arena_bytes']!=67108864 or d['tokens'] not in (128,256,512) or d['w8_mask']!=prep['config_w8_mask'] or d['threshold_bits']!=int(prep['threshold_bits_hex'],16):raise ValueError('Frozen model/deployment mismatch')
 for name,key in [('token_ids','token_offset'),('type_ids','type_offset'),('attention_mask','mask_offset')]:
  if prep['written_buffers'][name]['offset']!=d[key]:raise ValueError('Input layout mismatch')
 if prep['written_buffers']['token_ids']['bytes']!=4*d['tokens']:raise ValueError('Length mismatch')
 record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),state='preview',passed=False,length=d['tokens'],model_id=d['model_id'],precision_map=prep['precision_map'],w8_mask=d['w8_mask'],threshold_bits=d['threshold_bits'],deployment=d,arena_record=str(arena_record),deployment_job=str(resolve(a.deployment_job)),sd_metadata_lba=d['arena_sd_lba'],sd_arena_lba=d['arena_sd_lba']+1,sd_slot_bytes=512+67108864,physical_allocation_confirmed=False,physical_board_activity=False,physical_runtime=False,boot_packaged=False,integrated_eligible=False,integrated_closure=None)
 record.update(mode=prep.get('mode'),purpose=prep.get('purpose'),model_identity=job.get('model_identity',prep.get('model_identity')),development_freeze=job.get('development_freeze'),numerical_acceptance=prep.get('numerical_acceptance'),threshold_scope=prep.get('threshold_scope'),threshold_transfer=prep.get('threshold_transfer'),threshold_source_length=prep.get('threshold_source_length'),threshold_deployed_length=prep.get('threshold_deployed_length'))
 if a.system_dir or a.host_link_dir or a.bootgen:
  if not a.system_dir or not a.host_link_dir:raise ValueError('Boot packaging needs both actual implemented system and linked host')
  system=resolve(a.system_dir);host=resolve(a.host_link_dir);sr=load(system/'run.json');hr=load(host/'run.json')
  if a.implementation_record:
   resumed=load(resolve(a.implementation_record))
   if resumed.get('passed') is not True or Path(resumed['original_run']).resolve()!=(system/'run.json').resolve():raise ValueError('Resume must be an actually successful run of this same frozen project')
   sr=dict(sr,passed=True);record['implementation_record']=str(resolve(a.implementation_record))
  if sr.get('passed') is not True or sr['phase']!='implement' or sr['length']!=d['tokens'] or hr.get('passed') is not True:raise ValueError('Actual completed implementation and linked host required')
  xsa=system/'sentry_vek280.xsa';elf=host/'supervisor.elf'
  pdis=list((system/'project/sentry_vek280.runs/impl_1').glob('*.pdi'))
  if not xsa.is_file() or not elf.is_file() or len(pdis)!=1:raise ValueError('Need exact implemented PDI/XSA and supervisor ELF')
  # BSP/ELF sharing is permitted only for the same verified PS/PL apertures.
  rows=[line.split('\t') for line in (system/'address_segments.tsv').read_text().splitlines()]
  shared=[r for r in rows if 'DDR_LOW0' in r[1] and r[2]];mail=[r for r in rows if r[0]=='/cips/M_AXI_FPD' and 'mailbox' in r[1]]
  if len(shared)!=16 or len(mail)!=1 or int(mail[0][2],0)!=d['mailbox_base'] or any(not int(r[2],0)<=d['arena_base'] or d['arena_base']+d['arena_bytes']>int(r[2],0)+int(r[3],0) for r in shared):raise ValueError('Current XSA aperture differs from linked deployment')
  previous=load(resolve(hr['deployment_job']) if hr.get('deployment_job') else Path(hr['prior_host_run']).parent/'job.json')['selected_deployment']
  if previous!=d:raise ValueError('ELF was compiled for a different deployment')
  record.update(profile_id=f"lanes{sr['lanes']}",system_run=str(system/'run.json'),host_run=str(host/'run.json'),pdi=str(pdis[0]),xsa=str(xsa),elf=str(elf),timer_frequency_hz=hr.get('timer_frequency_hz'),timing_resolution_ns=1e9/hr['timer_frequency_hz'] if hr.get('timer_frequency_hz') else None)
  if a.integrated_closure:
   closure=load(resolve(a.integrated_closure));config=load(ROOT/'configs/project.json');limits=config['hardware']['resource_limits'];used=closure.get('resources',{})
   if (closure.get('part')!=config['hardware']['part'] or closure.get('profile_id')!=record['profile_id'] or closure.get('length')!=d['tokens'] or resolve(closure['system_directory']).resolve()!=system.resolve() or closure.get('integrated_eligible') is not True or closure.get('timing_met') is not True or closure.get('bandwidth_feasible') is not True or set(used)!=set(limits) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 or v>limits[k] for k,v in used.items())):raise ValueError('Actual integrated closure does not qualify this deployment')
   from inspect_vek280_system import inspect_system
   reparsed,_,_=inspect_system(system,a.implementation_record,config)
   fields=('part','profile_id','length','system_directory','source_snapshot','run_record','resources','timing_met','bandwidth_feasible','integrated_eligible','drc_errors_or_critical_warnings','wns_ns','whs_ns')
   if reparsed['integrated_eligible'] is not True or any(closure.get(k)!=reparsed[k] for k in fields):raise ValueError('Closure differs from independently reparsed actual routed system')
   record.update(integrated_eligible=True,integrated_closure=str(resolve(a.integrated_closure).resolve().relative_to(ROOT.resolve())),integrated_closure_reparsed=True)
 elif a.integrated_closure:raise ValueError('Integrated closure requires its implemented system and host')
 if not a.execute:print(json.dumps(record,indent=2));return
 if out.exists():raise ValueError('Preserve prior package; choose fresh output')
 out.mkdir(parents=True);record['state']='packing'
 try:
  meta=bytearray(512);struct.pack_into('<8sIIQIII',meta,0,b'SENTRYA1',1,d['tokens'],d['model_id'],d['w8_mask'],d['threshold_bits'],d['arena_bytes'])
  with (out/'model_slot.bin').open('wb') as f:
   f.write(meta)
   with image.open('rb') as source:shutil.copyfileobj(source,f,1024*1024)
  (out/'model_metadata.bin').write_bytes(meta)
  if 'pdi' in record:
   shutil.copyfile(record['pdi'],out/'implemented.pdi');shutil.copyfile(record['elf'],out/'supervisor.elf')
   # AMD UG1283: import Vivado bootimage and add A72 secure EL3 application.
   bif='sentry_boot:\n{\n image { { type=bootimage, file=implemented.pdi } }\n image { name=apu_ss, id=0x1c000000\n  { core=a72-0, exception_level=el-3, trustzone, file=supervisor.elf }\n }\n}\n'
   (out/'boot.bif').write_text(bif)
   command=['taskset','-c','0-5','/home/jp19126/Xilinx/2025.2/Vivado/bin/bootgen','-arch','versal','-image','boot.bif','-o','BOOT.BIN','-w','on'];record['bootgen_command']=command
   if a.bootgen:
    if os.name=='nt':raise RuntimeError('Offline Bootgen runs with WSL Linux Python')
    if (ROOT/'results/goal2/pause.request').exists():raise RuntimeError('Pause marker present')
    r=subprocess.run(command,cwd=out,text=True,capture_output=True,timeout=120);(out/'bootgen.log').write_text(r.stdout+r.stderr)
    if r.returncode or not (out/'BOOT.BIN').is_file():raise RuntimeError('Bootgen failed; inspect log')
    record['boot_packaged']=True
  record.update(state='packed',passed=True,model_slot=str(out/'model_slot.bin'))
 except BaseException as exc:record.update(state='failed',error=str(exc));raise
 finally:(out/'package.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(output=str(out),**record),indent=2))
if __name__=='__main__':main()
