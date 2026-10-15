"""Bounded preparation/CPU replay and explicit vendor dispatch for full CNN.

Run only on coordinator-assigned resources. `prepare` never invokes a vendor.
Inputs/outputs are explicit: worktree source and canonical artifacts stay apart.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,signal,struct,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def linux(p):
 s=str(Path(p).resolve()).replace('\\','/')
 if s.startswith('//wsl.localhost/Ubuntu/'):return '/'+s.split('/Ubuntu/',1)[1]
 if len(s)>2 and s[1]==':':return '/mnt/'+s[0].lower()+s[2:]
 return s
def command(args):return ['wsl','--exec',*args] if os.name=='nt' else args
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def fixtures(out):
 import numpy as np,torch
 from numerical import CONFIG,calibrate,export_model,IntegerCNN,fake_quant_model,shift,divide
 sys.path.insert(0,str(ROOT/'scripts'));from cnn_models import build_model
 torch.set_num_threads(2);torch.manual_seed(271828)
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 m=build_model(CONFIG,30522).eval();ids=torch.randint(1,30522,(5,256));mask=torch.ones_like(ids)
 mask[0,7:]=0;mask[1,255:]=0;mask[2,4::5]=0;mask[3]=0
 ids[~mask.bool()]=65535 # Padding IDs must never be dereferenced.
 calibration_ids=ids.masked_fill(~mask.bool(),0)
 e=calibrate(m,calibration_ids,mask);export_model(m,e,out/'export',{'kind':'deterministic untrained synthetic fixture','seed':271828,'no_quality_evidence':True})
 ref=IntegerCNN(out/'export')
 with torch.inference_mode():margins=ref.margin(ids,mask).tolist()
 rows=[]
 def row(index,first=1,final=1,previous=0,previous_windows=0,threshold=0):
  maximum=max([margins[i] for i in index]+([] if first else [previous]))
  rows.append(dict(indices=index,first=first,final=final,previous=previous,previous_windows=previous_windows,threshold=threshold,maximum=maximum,blocked=int(final and maximum>threshold)))
 for i in range(5):row([i])
 row([0,1],final=0);row([2,4],first=0,previous=max(margins[0:2]),previous_windows=2)
 row([4],threshold=margins[4]);row([4],threshold=margins[4]-1)
 pack_cases(out/'cases.bin',rows,ids.numpy(),mask.numpy())
 # Representative RTL: short, full, multiwindow, continued and equality.
 selected=[rows[i] for i in [0,4,5,6,7]]
 pack_cases(out/'cases_rtl.bin',selected,ids.numpy(),mask.numpy())
 write_json(out/'cases.json',{'rows':rows,'margins':margins,'mask_lengths':mask.sum(1).tolist(),'kind':'synthetic full graph arithmetic fixtures'})
 # Training adapter should propagate a real nonzero gradient without a teacher.
 fake=fake_quant_model(m,e);fake.train();value=fake(calibration_ids[:2],mask[:2]).logits;value.square().sum().backward()
 grad=float(m.classifier.weight.grad.abs().sum());assert grad>0 and torch.isfinite(m.classifier.weight.grad).all()
 rng=torch.get_rng_state()
 with torch.no_grad():m(calibration_ids[:2],mask[:2])
 expected_rng=torch.get_rng_state();torch.set_rng_state(rng)
 with torch.no_grad():fake(calibration_ids[:2],mask[:2])
 assert torch.equal(expected_rng,torch.get_rng_state()),'QAT dropout RNG call/order mismatch'
 with torch.inference_mode():fy=fake.eval()(calibration_ids,mask).logits;iy=ref(calibration_ids,mask).logits
 # Scalar rounding check independent of vectorized tensor implementation.
 checks=0
 for s in [-62,-12,-3,-1,0,1,7,31]:
  for x in [-99999,-129,-127,-5,-3,-1,0,1,3,5,127,129,99999]:
   expected=max(-127,min(127,round(x*(2.**s))))
   assert int(shift(torch.tensor(x),s,127))==expected,(x,s);checks+=1
 for d in [1,2,3,4,255,256]:
  for x in range(-511,512):assert int(divide(torch.tensor(x),torch.tensor(d)))==round(x/d);checks+=1
 write_json(out/'fixture_check.json',{'rounding_cases':checks,'gradient_l1':grad,'dropout_rng_state_matches_base':True,'fake_exact_max_abs_logit':float((fy-iy).abs().max()),'fake_exact_equal_logits':int((fy==iy).sum()),'total_logits':fy.numel(),'note':'STE float reductions can differ at ties; exported exact path is definitive'})
 print(json.dumps({'fixture':str(out),'margins':margins,'exponents':e,'gradient_l1':grad}))

def pack_cases(path,rows,ids,masks):
 import numpy as np
 parts=[struct.pack('<I',len(rows))]
 for r in rows:
  parts.append(struct.pack('<IIIiIiii',len(r['indices']),r['first'],r['final'],r['previous'],r['previous_windows'],r['threshold'],r['maximum'],r['blocked']))
  parts.append(np.asarray(ids[r['indices']],dtype='<u2').tobytes());parts.append(np.asarray(masks[r['indices']],dtype='u1').tobytes())
 Path(path).write_bytes(b''.join(parts))

def real_cases(out,export,prepared):
 """Select by document size only; preserve complete documents and every tail."""
 import numpy as np,torch
 from numerical import IntegerCNN,fit_manifest
 prepared=Path(prepared);out=Path(out)
 fit_identity=fit_manifest(prepared,('ids.npy','masks.npy','starts.npy'))
 out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 ids=np.load(prepared/'ids.npy',mmap_mode='r');mask=np.load(prepared/'masks.npy',mmap_mode='r');starts=np.load(prepared/'starts.npy',mmap_mode='r')
 counts=np.diff(starts);single=np.flatnonzero(counts==1);multi=np.flatnonzero(counts>=2)
 selected={}
 if len(single):
  lens=np.asarray(mask[np.asarray(starts[single],dtype=np.int64)]).sum(1)
  selected['short_complete']=int(single[np.argmin(lens)]);selected['full_complete']=int(single[np.argmax(lens)])
 if len(multi):selected['multi_complete']=int(multi[np.argmin(counts[multi])])
 if not selected:raise ValueError('No complete prepared documents')
 chosen=list(dict.fromkeys(selected.values()));all_indices=[];ranges={}
 for di in chosen:
  n=int(counts[di]);ranges[di]=(len(all_indices),len(all_indices)+n);all_indices.extend(range(int(starts[di]),int(starts[di+1])))
 xi=np.asarray(ids[all_indices],dtype=np.int64);ma=np.asarray(mask[all_indices],dtype=np.int64)
 ref=IntegerCNN(export);margins=[]
 with torch.inference_mode():
  for i in range(len(xi)):margins.append(int(ref.margin(torch.tensor(xi[i:i+1]),torch.tensor(ma[i:i+1]))[0]))
 rows=[]
 for di in chosen:
  lo,hi=ranges[di];previous=-(1<<31);windows=0
  # A first one-window chunk followed by remaining chunks exercises continuation.
  at=lo
  while at<hi:
   end=min(hi,at+(1 if at==lo else 16));index=list(range(at,end));maximum=max([previous]+[margins[i] for i in index]);final=int(end==hi)
   rows.append(dict(indices=index,first=int(at==lo),final=final,previous=0 if at==lo else previous,previous_windows=windows,threshold=0,maximum=maximum,blocked=int(final and maximum>0),document_index=di))
   previous=maximum;windows+=len(index);at=end
 pack_cases(out/'cases.bin',rows,xi,ma);pack_cases(out/'cases_rtl.bin',rows,xi,ma)
 write_json(out/'cases.json',{'selection':selected,'window_indices':all_indices,'window_margins_q12':margins,'rows':rows,'prepared':str(prepared),'export_manifest_sha256':digest(Path(export)/'manifest.json'),'fit_identity':fit_identity,'boundary':'FIT inputs only; complete selected documents; no labels or quality metrics'})
 print(json.dumps({'real_cases':str(out),'documents':len(chosen),'windows':len(all_indices),'calls':len(rows)}))

def prepare(out,export,cases,lanes):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);src=out/'source';src.mkdir()
 for name in ['cnn.hpp','cnn.cpp','testbench.cpp']:shutil.copyfile(ROOT/'hardware/kd_cnn_v2'/name,src/name)
 shutil.copyfile(Path(export)/'model.bin',out/'model.bin');shutil.copyfile(cases,out/'cases.bin')
 rtl=Path(cases).with_name('cases_rtl.bin');shutil.copyfile(rtl if rtl.exists() else cases,out/'cases_rtl.bin')
 flags=f'-std=c++17 -DCNN_LANES={lanes}'
 base=[f'open_project {{{linux(out/"project")}}}','set_top kd_cnn_top',
  f'add_files {{{linux(src/"cnn.cpp")}}} -cflags {{{flags}}}',f'add_files -tb {{{linux(src/"testbench.cpp")}}} -cflags {{{flags}}}',
  'open_solution solution -flow_target vivado','set_part xcve2802-vsvh1760-2MP-e-S','create_clock -period 5.0 -name default','set_clock_uncertainty 0.5']
 for arg in ['blob','bytes','ids','masks','count','load','first','final','previous','previous_windows','threshold','output','return']:
  base.append(f'set_directive_interface -mode s_axilite -bundle control kd_cnn_top {arg}')
 for action in ['csim','synthesize','cosim','export']:
  tcl=list(base)
  if action in ['csim','synthesize']:tcl.append(f'csim_design -argv {{{linux(out/"model.bin")} {linux(out/"cases.bin")}}}')
  if action=='synthesize':tcl.append('csynth_design')
  if action=='cosim':tcl.append(f'cosim_design -rtl verilog -tool xsim -trace_level none -argv {{{linux(out/"model.bin")} {linux(out/"cases_rtl.bin")}}}')
  if action=='export':tcl.append('export_design -format ip_catalog -rtl verilog -vendor sentry -library research -version 1.0')
  (out/(action+'.tcl')).write_text('\n'.join(tcl+['exit'])+'\n',encoding='utf-8')
 record={'lanes':lanes,'source_hashes':{p.name:digest(p) for p in src.iterdir()},'model_sha256':digest(out/'model.bin'),'cases_sha256':digest(out/'cases.bin'),'cases_rtl_sha256':digest(out/'cases_rtl.bin'),'evidence':'prepared only','source_root':str(ROOT.resolve()),'target':'xcve2802-vsvh1760-2MP-e-S','period_ns':5.,'uncertainty_ns':.5}
 write_json(out/'prepared.json',record);print(json.dumps(record))

def cpu(out):
 out=Path(out);r=json.loads((out/'prepared.json').read_text());exe=out/'cnn_reference';log=out/'cpu.stdout.log'
 if log.exists():raise FileExistsError('Preserve prior CPU evidence')
 compile_args=command(['g++','-std=c++17','-O2','-Wno-unknown-pragmas',f'-DCNN_LANES={r["lanes"]}',linux(out/'source/cnn.cpp'),linux(out/'source/testbench.cpp'),'-o',linux(exe)])
 start=time.monotonic();p=subprocess.run(compile_args,capture_output=True,text=True,timeout=120)
 write_json(out/'c_build.json',{'command':compile_args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.monotonic()-start,'source_hashes':r['source_hashes']})
 p.check_returncode();args=command([linux(exe),linux(out/'model.bin'),linux(out/'cases.bin')]);start=time.monotonic()
 p=subprocess.run(args,capture_output=True,text=True,timeout=120);log.write_text(p.stdout+p.stderr,encoding='utf-8')
 write_json(out/'c_run.json',{'command':args,'returncode':p.returncode,'seconds':time.monotonic()-start,'executable_sha256':digest(exe),'boundary':'native C functional replay; not performance estimate'})
 print(p.stdout);p.check_returncode()

def prepare_cosim(out):
 """Repair only verification-buffer extents, preserving synthesized RTL/source."""
 out=Path(out).resolve();record=out/'cosim_testbench.json'
 if record.exists():raise FileExistsError('Co-sim testbench already frozen')
 tb=out/'source/testbench_cosim.cpp';shutil.copyfile(ROOT/'hardware/kd_cnn_v2/testbench.cpp',tb)
 r=json.loads((out/'prepared.json').read_text());args=command(['g++','-std=c++17','-O2','-Wno-unknown-pragmas',f'-DCNN_LANES={r["lanes"]}',linux(out/'source/cnn.cpp'),linux(tb),'-o',linux(out/'cnn_cosim_reference')])
 p=subprocess.run(args,capture_output=True,text=True,timeout=120);p.check_returncode()
 run=command([linux(out/'cnn_cosim_reference'),linux(out/'model.bin'),linux(out/'cases.bin')]);p=subprocess.run(run,capture_output=True,text=True,timeout=120)
 write_json(record,{'compile_command':args,'run_command':run,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'testbench_sha256':digest(tb),'kernel_sha256':digest(out/'source/cnn.cpp'),'reason':'Allocate declared AXI serialization extents for all short input/model pointers; no kernel/arithmetic change'})
 p.check_returncode();script=out/'cosim.tcl';shutil.copyfile(script,out/'cosim.pre_extent_fix.tcl');tcl=script.read_text()
 before='cosim_design '
 tcl=tcl.replace(before,f'remove_files {{{linux(out/"source/testbench.cpp")}}}\nadd_files -tb {{{linux(tb)}}} -cflags {{-std=c++17 -DCNN_LANES={r["lanes"]}}}\n'+before)
 script.write_text(tcl);print(p.stdout)

def prepare_runtime(out,export,cases):
 out=Path(out).resolve();export=Path(export).resolve();cases=Path(cases).resolve();record=out/'cosim_runtime.json'
 if record.exists():raise FileExistsError('Preserve existing runtime identity')
 manifest=json.loads((export/'manifest.json').read_text())
 if digest(export/'model.bin')!=manifest['model_sha256']:raise ValueError('Model export changed')
 shutil.copyfile(export/'model.bin',out/'runtime_model.bin');shutil.copyfile(cases,out/'runtime_cases.bin')
 args=command([linux(out/'cnn_cosim_reference'),linux(out/'runtime_model.bin'),linux(out/'runtime_cases.bin')]);p=subprocess.run(args,capture_output=True,text=True,timeout=120)
 write_json(record,{'export':str(export),'export_manifest_sha256':digest(export/'manifest.json'),'model_sha256':digest(out/'runtime_model.bin'),'cases_source':str(cases),'cases_sha256':digest(out/'runtime_cases.bin'),'testbench_sha256':digest(out/'source/testbench_cosim.cpp'),'native_command':args,'native_returncode':p.returncode,'native_stdout':p.stdout,'native_stderr':p.stderr,'boundary':'Trained runtime weights and realFIT cases; unchanged synthesized kernel'})
 p.check_returncode();script=out/'cosim.tcl';shutil.copyfile(script,out/'cosim.synthetic_runtime.tcl')
 tcl=script.read_text().replace(linux(out/'model.bin'),linux(out/'runtime_model.bin')).replace(linux(out/'cases_rtl.bin'),linux(out/'runtime_cases.bin'))
 script.write_text(tcl);print(p.stdout)

def retry_cosim(out):
 """Preserve a failed full testbench attempt; retry representative valid calls."""
 out=Path(out).resolve();failed=json.loads((out/'cosim_run.json').read_text())
 if failed['state']!='failed':raise ValueError('Only failed attempts may be archived')
 archive=out/'cosim_attempt1';archive.mkdir(exist_ok=False)
 for name in ['cosim_run.json','cosim.stdout.log','cosim.resources.jsonl','cosim.tcl','project/solution/sim']:
  src=out/name
  if src.exists():shutil.move(str(src),str(archive/src.name))
 tb=out/'source/testbench_cosim_representative.cpp'
 text=(out/'source/testbench_cosim.cpp').read_text();text=text.split(' // Failure/reset behavior')[0]+' std::cout<<"ROUNDING "<<rounding<<" INTERFACE native-only\\n";return 0;\n}\n'
 tb.write_text(text)
 r=json.loads((out/'prepared.json').read_text());exe=out/'cnn_cosim_representative'
 args=command(['g++','-std=c++17','-O2','-Wno-unknown-pragmas',f'-DCNN_LANES={r["lanes"]}',linux(out/'source/cnn.cpp'),linux(tb),'-o',linux(exe)])
 p=subprocess.run(args,capture_output=True,text=True,timeout=120);p.check_returncode()
 run=command([linux(exe),linux(out/'runtime_model.bin'),linux(out/'runtime_cases.bin')]);p=subprocess.run(run,capture_output=True,text=True,timeout=120)
 write_json(out/'cosim_retry.json',{'source':str(tb),'sha256':digest(tb),'compile_command':args,'run_command':run,'returncode':p.returncode,'stdout':p.stdout,'reason':'Reduce serialized AXI transaction memory; retain four realFIT calls and native interface checks separately'})
 p.check_returncode()
 tcl=(archive/'cosim.tcl').read_text();tcl=tcl.replace('add_files -tb {'+linux(out/'source/testbench_cosim.cpp')+'}', 'remove_files {'+linux(out/'source/testbench_cosim.cpp')+'}\nadd_files -tb {'+linux(tb)+'}')
 (out/'cosim.tcl').write_text(tcl);print(p.stdout)


def host(out,export,cases,register_header=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
 names=['cnn.hpp','cnn.cpp','host_driver.hpp','host_driver.cpp','host_test.cpp']
 for n in names:shutil.copyfile(ROOT/'hardware/kd_cnn_v2'/n,source/n)
 flags=[]
 if register_header:
  shutil.copyfile(register_header,source/'synthesized_register_map.hpp');names.append('synthesized_register_map.hpp');flags=['-DCNN_REGISTER_MAP_HEADER="synthesized_register_map.hpp"']
 args=command(['g++','-std=c++17','-O2','-Wno-unknown-pragmas','-DCNN_LANES=16',*flags,linux(source/'cnn.cpp'),linux(source/'host_driver.cpp'),linux(source/'host_test.cpp'),'-o',linux(out/'host_test')])
 start=time.monotonic();p=subprocess.run(args,capture_output=True,text=True,timeout=120)
 write_json(out/'build.json',{'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.monotonic()-start,'source_hashes':{n:digest(source/n) for n in names}});p.check_returncode()
 args=command([linux(out/'host_test'),linux(Path(export)/'model.bin'),linux(cases)]);start=time.monotonic();p=subprocess.run(args,capture_output=True,text=True,timeout=120)
 write_json(out/'run.json',{'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'seconds':time.monotonic()-start,'model_sha256':digest(Path(export)/'model.bin'),'cases_sha256':digest(cases)});print(p.stdout);p.check_returncode()

def host_compile(out,bsp):
 if os.name!='posix':raise RuntimeError('Cross-compile from WSL')
 bsp=Path(bsp).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=False)
 if not (bsp/'include/xparameters.h').exists():raise FileNotFoundError('Explicit completed BSP required')
 names=['cnn.hpp','host_driver.hpp','host_driver.cpp','host_a72.hpp','host_a72.cpp']
 for n in names:shutil.copyfile(ROOT/'hardware/kd_cnn_v2'/n,out/n)
 compiler='/home/jp19126/Xilinx/2025.2/gnu/aarch64/lin/aarch64-none/bin/aarch64-none-elf-g++'
 records=[]
 for source in ['host_driver.cpp','host_a72.cpp']:
  args=[compiler,'-std=c++17','-O2','-fno-exceptions','-fno-rtti','-DSDT','-DARMA72_EL3','-I'+str(bsp/'include'),'-I'+str(out),'-c',str(out/source),'-o',str(out/(source+'.o'))]
  p=subprocess.run(args,capture_output=True,text=True,timeout=60);records.append({'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
  write_json(out/'compile.json',{'commands':records,'bsp':str(bsp),'bsp_xparameters_sha256':digest(bsp/'include/xparameters.h'),'source_hashes':{n:digest(out/n) for n in names},'boundary':'A72 object compatibility only with named existing BSP; not a newly connected XSA/ELF or physical execution'});p.check_returncode()
 print(json.dumps({'a72_objects':str(out),'passed':True}))

def vendor(out,action,seconds):
 if os.name!='posix':raise RuntimeError('Vendor wrapper requires WSL python3')
 out=Path(out).resolve();r=json.loads((out/'prepared.json').read_text());recordpath=out/(action+'_run.json')
 if recordpath.exists():raise FileExistsError('Keep existing run; resume must name a fresh attempt')
 if not 1<=seconds<=7200:raise ValueError('Bound vendor execution to <=7200 seconds')
 if json.loads((out/'c_run.json').read_text())['returncode']!=0:raise RuntimeError('CPU check required')
 for name,h in r['source_hashes'].items():
  if digest(out/'source'/name)!=h:raise ValueError('Source snapshot changed')
 for name,key in [('model.bin','model_sha256'),('cases.bin','cases_sha256'),('cases_rtl.bin','cases_rtl_sha256')]:
  if digest(out/name)!=r[key]:raise ValueError('Prepared artifact changed')
 if action=='cosim' and (out/'cosim_retry.json').exists():
  retry=json.loads((out/'cosim_retry.json').read_text())
  if digest(Path(retry['source']))!=retry['sha256'] or retry['returncode']!=0:raise ValueError('Retry testbench qualification changed')
 if action=='cosim' and (out/'cosim_runtime.json').exists():
  rt=json.loads((out/'cosim_runtime.json').read_text())
  for name,key in [('runtime_model.bin','model_sha256'),('runtime_cases.bin','cases_sha256'),('source/testbench_cosim.cpp','testbench_sha256')]:
   if digest(out/name)!=rt[key]:raise ValueError('Co-sim runtime changed')
  if rt['native_returncode']!=0:raise ValueError('Native runtime qualification failed')
 sys.path.insert(0,str(ROOT/'scripts'));from r1_challengers import resource_check
 pre=resource_check(True);args=['bash','-c','source "$1" && exec "$2" --mode hls --tcl "$3"','cnn-hls','/home/jp19126/Xilinx/2025.2/Vitis/settings64.sh','/home/jp19126/Xilinx/2025.2/Vitis/bin/vitis-run',str(out/(action+'.tcl'))]
 rec={'command':args,'started_utc':datetime.now(timezone.utc).isoformat(),'budget_seconds':seconds,'preflight':pre,'state':'running'}
 start=time.monotonic();proc=None
 try:
  with (out/(action+'.stdout.log')).open('x') as log:
   proc=subprocess.Popen(args,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=proc.pid;write_json(recordpath,rec)
   while proc.poll() is None:
    if (out/'STOP_REQUESTED').exists():raise InterruptedError('Coordinator stop requested; preserve vendor checkpoints')
    if time.monotonic()-start>seconds:raise TimeoutError('Vendor execution ceiling')
    obs=resource_check()
    with (out/(action+'.resources.jsonl')).open('a') as f:f.write(json.dumps(obs)+'\n')
    time.sleep(5)
   rec['returncode']=proc.returncode;rec['state']='completed' if proc.returncode==0 else 'failed'
 except BaseException as error:
  rec.update(state='failed',error=repr(error));raise
 finally:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
  rec['elapsed_seconds']=time.monotonic()-start;write_json(recordpath,rec)
 if proc.returncode:raise RuntimeError('Vendor failed; see retained log')

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['fixture','cases','prepare','prepare-cosim','prepare-runtime','retry-cosim','cpu','host','host-compile','csim','synthesize','cosim','export']);p.add_argument('--out',type=Path,required=True);p.add_argument('--export',type=Path);p.add_argument('--prepared',type=Path);p.add_argument('--bsp',type=Path);p.add_argument('--cases',type=Path);p.add_argument('--register-header',type=Path);p.add_argument('--lanes',type=int,choices=[4,16],default=16);p.add_argument('--seconds',type=int,default=1800)
 a=p.parse_args()
 if a.action=='fixture':fixtures(a.out)
 elif a.action=='cases':real_cases(a.out,a.export,a.prepared)
 elif a.action=='prepare':prepare(a.out,a.export,a.cases,a.lanes)
 elif a.action=='prepare-cosim':prepare_cosim(a.out)
 elif a.action=='prepare-runtime':prepare_runtime(a.out,a.export,a.cases)
 elif a.action=='retry-cosim':retry_cosim(a.out)
 elif a.action=='cpu':cpu(a.out)
 elif a.action=='host':host(a.out,a.export,a.cases,a.register_header)
 elif a.action=='host-compile':host_compile(a.out,a.bsp)
 else:vendor(a.out,a.action,a.seconds)
if __name__=='__main__':main()
