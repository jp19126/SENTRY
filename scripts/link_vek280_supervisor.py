"""Link a corrected supervisor against an already successful selected-XSA BSP."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,re,shutil,subprocess,struct
ROOT=Path(__file__).resolve().parents[1]
def inspect_deployment_elf(path,symbols,expected):
 """Read this supervisor's fixed AArch64 deployment constant from actual ELF bytes."""
 data=path.read_bytes()
 if data[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<H',data,18)[0]!=183:raise ValueError('Actual AArch64 little-endian64 ELF required')
 m=re.search(r'^([0-9a-f]+)\s+([0-9a-f]+)\s+R\s+deployment$',symbols,re.M)
 if not m or int(m[2],16)!=80:raise ValueError('Actual80-byte deployment symbol missing')
 address=int(m[1],16);phoff=struct.unpack_from('<Q',data,32)[0];size,num=struct.unpack_from('<HH',data,54);raw=None
 for i in range(num):
  typ,flags,offset,vaddr,paddr,filesz,memsz,align=struct.unpack_from('<II6Q',data,phoff+i*size)
  if typ==1 and vaddr<=address and address+80<=vaddr+filesz:raw=data[offset+address-vaddr:offset+address-vaddr+80]
 if raw is None:raise ValueError('Deployment symbol outside actual file-backed LOAD region')
 names=['mailbox_base','arena_base','uart_base','sd_base','model_id','mailbox_poll_limit','tokens','arena_bytes','arena_sd_lba','token_offset','type_offset','mask_offset','threshold_bits','w8_mask']
 actual=dict(zip(names,struct.unpack('<6Q7IH2x',raw)))
 if any(actual[k]!=v for k,v in expected.items()):raise ValueError('Actual compiled deployment differs from selected job or BSP peripherals')
 return dict(symbol_address=address,symbol_bytes=80,actual_embedded_deployment=actual)


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--host-run',type=Path,required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--deployment-job',type=Path,help='Generated matching frozen model/length deployment job; reuse only the verified same PS BSP');a=ap.parse_args()
 prior=a.host_run if a.host_run.is_absolute() else ROOT/a.host_run;bsp_job=json.loads((prior/'job.json').read_text());workspace=Path(bsp_job['workspace'])
 deployment_job=(a.deployment_job if a.deployment_job.is_absolute() else ROOT/a.deployment_job) if a.deployment_job else prior/'job.json'
 job=json.loads(deployment_job.read_text());deployment=job['selected_deployment']
 if deployment['tokens'] not in (128,256,512) or deployment['arena_bytes']!=67108864 or deployment['mailbox_base']!=bsp_job['selected_deployment']['mailbox_base'] or not 0<=deployment['w8_mask']<=65535:raise ValueError('Unsupported frozen deployment or changed BSP mailbox aperture')
 bsp=workspace/'sentry_a72/export/sentry_a72/sw/sentry_standalone'
 if not (bsp/'lib/libxil.a').is_file() or 'Platform Build Finished successfully.' not in (workspace/'_ide/logs/vitis.log').read_text():raise ValueError('Completed selected-XSA BSP required')
 linker=workspace/'supervisor/src/lscript.ld';ld=linker.read_text()
 region=re.search(r'(\w*ocm\w*)\s*:\s*ORIGIN\s*=\s*(0x[\da-fA-F]+),\s*LENGTH\s*=\s*(0x[\da-fA-F]+)',ld)
 if not region or any(name!=region[1] for name in re.findall(r'>\s*(\w+)',ld)):raise ValueError('All actual program sections must map to the selected OCM region')
 ld=re.sub(r'(_STACK_SIZE\s*=.*?\? _STACK_SIZE : )0x[\da-fA-F]+',r'\g<1>0x2000',ld)
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out=ROOT/'build/vek280_host'/f'l{job["selected_deployment"]["tokens"]}'/(stamp+'_link');out.mkdir(parents=True)
 for name in ['main.c','sentry_supervisor.c','sentry_supervisor.h']:shutil.copyfile(ROOT/'hardware/host'/name,out/name)
 shutil.copyfile(Path(job['sources'])/'deployment.h',out/'deployment.h');(out/'lscript.ld').write_text(ld)
 compiler=Path('/home/jp19126/Xilinx/2025.2/gnu/aarch64/lin/aarch64-none/bin/aarch64-none-elf-gcc')
 flags=['-DSDT','-DARMA72_EL3','-std=c11','-O2','-g','-Wall','-Wextra','-Werror','-specs='+str(bsp/'Xilinx.spec'),'-I'+str(bsp/'include'),'-I'+str(out)]
 commands=[[str(compiler),*flags,'-c',str(out/(name+'.c')),'-o',str(out/(name+'.o'))] for name in ['main','sentry_supervisor']]
 commands += [[str(compiler),*flags,str(out/'main.o'),str(out/'sentry_supervisor.o'),'-o',str(out/'supervisor.elf'),'-Wl,-T,'+str(out/'lscript.ld'),'-Wl,-Map,'+str(out/'supervisor.map'),'-L'+str(bsp/'lib'),'-Wl,--start-group','-lxilstandalone','-lxiltimer','-lxil','-lgcc','-lc','-Wl,--end-group'],[str(compiler).replace('gcc','readelf'),'-lW',str(out/'supervisor.elf')],[str(compiler).replace('gcc','nm'),'-S',str(out/'supervisor.elf')]]
 parameters=(bsp/'include/xparameters.h').read_text()
 timer=re.search(r'#define XPAR_CPU_TIMESTAMP_CLK_FREQ (\d+)',parameters)
 if not timer:raise ValueError('Actual selected BSP timer frequency missing')
 record=dict(timer_frequency_hz=int(timer[1]),timer_source=str(bsp/'include/xtimer_config.h'),timer_timestamp_source='XTime_GetTime from selected xiltimer BSP; COUNTS_PER_SECOND=XPAR_CPU_TIMESTAMP_CLK_FREQ',response_kernel_interval_bytes=[56,63],timing_boundary='A72 mailbox submit through observed completion; excludes UART, input copies and cache flush; includes A72 control/poll overhead',passed=False,state='prepared',prior_host_run=str(prior/'run.json'),selected_xsa=bsp_job['xsa'],deployment_job=str(deployment_job),selected_deployment=deployment,bsp=str(bsp),ocm=dict(name=region[1],base=int(region[2],0),bytes=int(region[3],0)),stack_bytes=8192,sd_bounce_bytes=16384,commands=commands,board_access=False,physical_runtime=False)
 def save():(out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
 save();print(json.dumps(dict(output=str(out),**record),indent=2),flush=True)
 if not a.execute:return
 if os.name=='nt':raise RuntimeError('Use WSL Linux Python')
 try:
  for i,command in enumerate(commands):
   if (ROOT/'results/goal2/pause.request').exists():raise RuntimeError('Pause marker present')
   result=subprocess.run(command,cwd=out,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
   (out/f'command_{i}.stdout.log').write_text(result.stdout)
   if result.returncode:raise RuntimeError(f'Host compile/link command{i} failed')
  headers=(out/'command_3.stdout.log').read_text();symbols=(out/'command_4.stdout.log').read_text()
  loads=[]
  for line in headers.splitlines():
   if line.strip().startswith('LOAD'):
    fields=line.split();base=int(fields[3],16);size=int(fields[5],16)
    if not record['ocm']['base']<=base or base+size>record['ocm']['base']+record['ocm']['bytes']:raise ValueError('ELF load outside actual OCM')
    loads.append(dict(base=base,memory_bytes=size))
  if not loads:raise ValueError('Missing ELF LOAD segments')
  bounce=re.search(r'^([\da-f]+)\s+([\da-f]+)\s+\w\s+sd_chunk$',symbols,re.M)
  if not bounce or int(bounce[2],16)!=16384:raise ValueError('Actual bounce placement missing')
  expected=dict(deployment)
  for name,macro in [('uart_base','XPAR_XUARTPSV_0_BASEADDR'),('sd_base','XPAR_XSDPS_0_BASEADDR')]:
   parameter=re.search(r'^#define '+macro+r'\s+(0x[0-9a-fA-F]+|[0-9]+)',parameters,re.M)
   if not parameter:raise ValueError('Actual BSP peripheral address missing: '+macro)
   expected[name]=int(parameter[1],0)
  compiled_deployment=inspect_deployment_elf(out/'supervisor.elf',symbols,expected)
  record.update(passed=True,state='linked',elf=str(out/'supervisor.elf'),load_segments=loads,sd_bounce_base=int(bounce[1],16),compiled_deployment=compiled_deployment)
 except BaseException as exc:record.update(state='failed',error=str(exc));raise
 finally:save()
if __name__=='__main__':main()
