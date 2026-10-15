"""Prepare/run post-implementation offline BSP/ELF/boot packaging.
Invoked within hardware_system's existing deadline and process-group guard.
No physical operations; generated images remain unqualified for deployment.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
BSP_SCRIPT=r"""import json,os,re
from pathlib import Path
import vitis
job=json.loads(Path(os.environ['SENTRY_KD_PACKAGE_JOB']).read_text());out=Path(job['package']);workspace=out/'workspace'
if workspace.exists():raise ValueError('Preserve existing package workspace')
client=vitis.create_client()
try:
 client.set_workspace(str(workspace))
 platform=client.create_platform_component(name='kd_cnn_a72',hw_design=job['xsa'],os='standalone',cpu='psv_cortexa72_0',domain_name='kd_cnn_standalone',no_boot_bsp=True)
 platform.build();platform.report()
 app=client.create_app_component(name='kd_cnn_guard',platform=client.find_platform_in_repos('kd_cnn_a72'),domain='kd_cnn_standalone',template='empty_application')
 linker=app.get_ld_script();regions=linker.get_memory_regions();choices=[r for r in regions if 'ocm' in r['name'].lower()]
 if len(choices)!=1:raise ValueError('Need one actual OCM linker region: '+str(regions))
 ocm=choices[0];linker.set_stack_size('0x8000');linker.set_heap_size('0x1000');text=Path(linker.file_path).read_text();known={r['name'] for r in regions}
 def bind(m):
  if m[1] not in known:raise ValueError('Unknown linker region '+m[1])
  return '> '+ocm['name']
 text=re.sub(r'>\s*(\w+)',bind,text);(out/'lscript.ld').write_text(text)
 bsp=workspace/'kd_cnn_a72/export/kd_cnn_a72/sw/kd_cnn_standalone'
 if not (bsp/'include/xparameters.h').exists():raise FileNotFoundError(bsp)
 (out/'bsp_result.json').write_text(json.dumps({'bsp':str(bsp),'xsa':job['xsa'],'ocm':ocm,'linker':str(out/'lscript.ld'),'physical_execution':False},indent=2)+'\n')
finally:vitis.dispose()
"""
def prepare(out,host,project_root):
 out=Path(out).resolve();host=Path(host).resolve();project_root=Path(project_root).resolve()
 if (out/'run.json').exists() or (out/'packaging_job.json').exists():raise FileExistsError('Preparation must precede dispatch')
 script=out/'package_scripts';script.mkdir();shutil.copyfile(__file__,script/'hardware_package.py');shutil.copyfile(ROOT/'scripts/kd_cnn_v2/hardware_host.py',script/'hardware_host.py');(script/'generate_bsp.py').write_text(BSP_SCRIPT)
 snapshot=out/'package_host_snapshot';shutil.copytree(host/'source',snapshot/'source');shutil.copyfile(host/'prepared.json',snapshot/'prepared.json')
 job={'system':str(out),'package':str(out/'offline_package'),'xsa':str(out/'implemented.xsa'),'pdi':str(project_root/'project/kd_cnn_system.runs/impl_1/cnn_system_wrapper.pdi'),'host_snapshot':str(snapshot),'host_prepared_sha256':digest(snapshot/'prepared.json'),'script_sha256':{p.name:digest(p) for p in script.iterdir()},'purpose':'Offline representative model package; final nomination needs constants/relink if different','physical_execution':False}
 save(out/'packaging_job.json',job)
 report=out/'preimage_reports.tcl';report.write_text('report_timing_summary -report_unconstrained -file {'+str(out/'preimage_timing.rpt')+'}\nreport_drc -file {'+str(out/'preimage_drc.rpt')+'}\nreport_route_status -file {'+str(out/'preimage_route.rpt')+'}\n')
 tcl=out/'system.tcl';text=tcl.read_text();text=text.replace('reset_run impl_1\n','reset_run impl_1\nset_property STEPS.WRITE_DEVICE_IMAGE.TCL.PRE {'+str(report)+'} [get_runs impl_1]\n');tcl.write_text(text)
 p=out/'prepared.json';meta=json.loads(p.read_text());meta.update(packaging_job_sha256=digest(out/'packaging_job.json'),system_tcl_sha256=digest(tcl),preimage_reports_sha256=digest(report),endpoint='write_device_image PDI + implemented XSA, then actual-XSA BSP, GNU guard ELF and Bootgen under same original deadline; no physical qualification claim');save(p,meta)

def execute(job_path):
 job_path=Path(job_path).resolve();job=json.loads(job_path.read_text());out=Path(job['package']);out.mkdir(exist_ok=False);script=job_path.parent/'package_scripts';record=out/'package.json';steps=[];start=time.monotonic()
 for name,expected in job['script_sha256'].items():
  if digest(script/name)!=expected:raise ValueError('Frozen packaging script changed')
 if digest(Path(job['host_snapshot'])/'prepared.json')!=job['host_prepared_sha256']:raise ValueError('Host association changed')
 for key in ['pdi','xsa']:
  if not Path(job[key]).is_file():raise FileNotFoundError(job[key])
 def run(args,env=None):
  before=time.monotonic();r=subprocess.run(args,text=True,capture_output=True,env=env);steps.append({'command':args,'seconds':time.monotonic()-before,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});save(record,{'state':'running','steps':steps});r.check_returncode()
 try:
  env=os.environ.copy();env['SENTRY_KD_PACKAGE_JOB']=str(job_path)
  run(['/home/jp19126/Xilinx/2025.2/Vitis/bin/vitis','-s',str(script/'generate_bsp.py')],env)
  result=json.loads((out/'bsp_result.json').read_text());host=out/'host';shutil.copytree(job['host_snapshot'],host);shutil.copyfile(out/'lscript.ld',host/'lscript.ld');meta=json.loads((host/'prepared.json').read_text());bsp=Path(result['bsp']);meta.update(bsp=str(bsp),bsp_xparameters_sha256=digest(bsp/'include/xparameters.h'),linker_sha256=digest(host/'lscript.ld'),boundary='Matching implemented-XSA BSP with qualification model constants');save(host/'prepared.json',meta)
  run([sys.executable,str(script/'hardware_host.py'),'link','--out',str(host)])
  shutil.copyfile(host/'kd_cnn_guard.elf',out/'kd_cnn_guard.elf');shutil.copyfile(job['pdi'],out/'implemented.pdi')
  bif='kd_cnn_boot:\n{\n image { { type=bootimage, file='+str(out/'implemented.pdi')+' } }\n image { name=apu_ss, id=0x1c000000\n  { core=a72-0, exception_level=el-3, trustzone, file='+str(out/'kd_cnn_guard.elf')+' }\n }\n}\n';(out/'boot.bif').write_text(bif)
  run(['/home/jp19126/Xilinx/2025.2/Vivado/bin/bootgen','-arch','versal','-image',str(out/'boot.bif'),'-o',str(out/'BOOT.BIN'),'-w','on'])
  if not (out/'BOOT.BIN').is_file():raise RuntimeError('Bootgen produced no BOOT.BIN')
  save(record,{'state':'completed','seconds':time.monotonic()-start,'steps':steps,'sha256':{name:digest(out/name) for name in ['BOOT.BIN','implemented.pdi','kd_cnn_guard.elf','boot.bif']},'xsa_sha256':digest(job['xsa']),'host_prepared_sha256':digest(host/'prepared.json'),'model_sha256':meta['model_sha256'],'threshold_q12':meta['threshold'],'physical_execution':False,'deployment_qualified':False,'boundary':'Offline matching-platform package only; achieved routed timing/DRC and final nomination require independent report, physical validation not performed'})
 except BaseException as e:save(record,{'state':'failed','seconds':time.monotonic()-start,'steps':steps,'error':repr(e),'physical_execution':False});raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--out',type=Path);p.add_argument('--host',type=Path);p.add_argument('--project-root',type=Path);p.add_argument('--job',type=Path);a=p.parse_args()
 if a.prepare:prepare(a.out,a.host,a.project_root)
 else:execute(a.job)
if __name__=='__main__':main()
