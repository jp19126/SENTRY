"""Inspect one actual routed SENTRY system; never launch vendor tools or access hardware."""
from pathlib import Path
from datetime import datetime,timezone
from zipfile import ZipFile
import argparse,json,re
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def resolve(value):
 p=Path(value);return p if p.is_absolute() else ROOT/p
def relative(path):return str(Path(path).resolve().relative_to(ROOT.resolve()))
def require(ok,message):
 if not ok:raise ValueError(message)
def match(pattern,text,label,flags=0):
 m=re.search(pattern,text,flags);require(m is not None,'Missing actual '+label);return m

def inspect_system(system, implementation_record=None, config=None):
 """Read actual completed routed reports and DCP; return facts without writes or stdout."""
 system=resolve(system);run=read(system/'run.json');actual=run;run_path=system/'run.json'
 config=read(ROOT/'configs/project.json') if config is None else config
 if implementation_record:
  run_path=resolve(implementation_record);actual=read(run_path)
  require(resolve(actual['original_run']).resolve()==(system/'run.json').resolve(),'Resume belongs to another frozen system')
 require(actual.get('passed') is True and actual.get('state')=='returned' and run.get('phase')=='implement','Actually completed implementation required')
 require(run['length'] in (128,256,512) and run['lanes'] in (16,32,64),'Unsupported actual build identity')
 source=system/'sources';require(source.is_dir() and (source/'hardware/gate_data_mover.sv').is_file(),'Frozen actual core source missing')
 timing=(system/'integrated_timing.rpt').read_text(encoding='utf-8');util=(system/'integrated_utilization.rpt').read_text(encoding='utf-8');drc=(system/'integrated_drc.rpt').read_text(encoding='utf-8')
 require('Design State      : Routed' in timing and 'Design State : Routed' in util and 'Design State : Fully Routed' in drc,'Reports are not actual final routed reports')
 require('xcve2802-vsvh1760-2MP-e-S' in util,'Actual report part differs')
 rows=match(r'WNS\(ns\).*?\n\s*[- ]+\n\s*([^\n]+)',timing,'routed timing summary',re.S)[1].split()
 require(len(rows)==12,'Unexpected timing summary columns')
 wns,tns=float(rows[0]),float(rows[1]);setup=int(rows[2]);whs=float(rows[4]);hold=int(rows[6]);pulse=int(rows[10])
 checks={name:int(match(r'checking '+name+r' \((\d+)\)',timing,name)[1]) for name in ('no_clock','unconstrained_internal_endpoints','loops','latch_loops')}
 period=float(match(r'^clk_pl_0\s+\{[^}]+\}\s+([\d.]+)',timing,'PL clock period',re.M)[1])
 require(abs(period-1000/config['hardware']['clock_objective_mhz'])<1e-6,'Actual PL period differs from adopted objective')
 resources={}
 for key,label in [('LUT_ELEMENTS','CLB LUTs'),('FLIPFLOPS','Register as Flip Flop'),('BLOCK_RAMS','Block RAM Tile'),('ULTRA_RAMS','URAM'),('DSP','DSP Slices')]:
  resources[key]=float(match(r'^\|\s*'+re.escape(label)+r'\s*\|\s*([\d.]+)\s*\|',util,label,re.M)[1])
  if resources[key].is_integer():resources[key]=int(resources[key])
 limits=config['hardware']['resource_limits'];within=set(resources)==set(limits) and all(v<=limits[k] for k,v in resources.items())
 dcp=system/'project/sentry_vek280.runs/impl_1/sentry_system_wrapper_routed.dcp'
 with ZipFile(dcp) as archive:
  ncr=[n for n in archive.namelist() if n.endswith('.ncr')];nts=[n for n in archive.namelist() if n.endswith('.nts')]
  require(len(ncr)==len(nts)==1,'Exact routed NoC compiler records required');solution=json.loads(archive.read(ncr[0]));traffic=json.loads(archive.read(nts[0]))
 paths=solution.get('Paths',[]);bandwidth=solution.get('SolutionType')=='OPTIMAL' and len(paths)==7 and all(type(p.get(k)) in (int,float) and p[k]>0 and p.get(k.replace('BW','AchievedBW'),-1)>=p[k] for p in paths for k in ('ReadBW','WriteBW'))
 errors=[line.strip() for line in drc.splitlines() if re.search(r'^\|[^|]+\|\s*(?:Error|Critical Warning)\s*\|',line)]
 timing_met=wns>=0 and setup==0 and whs>=0 and hold==0 and pulse==0 and not any(checks.values())
 pdis=list((system/'project/sentry_vek280.runs/impl_1').glob('*.pdi'));xsa=system/'sentry_vek280.xsa'
 require(len(pdis)==1 and pdis[0].stat().st_size>0 and xsa.is_file() and xsa.stat().st_size>0,'Actual final PDI/XSA missing')
 eligible=timing_met and within and bandwidth and not errors
 record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),state='routed_eligible' if eligible else 'routed_ineligible',part=config['hardware']['part'],profile_id=f"lanes{run['lanes']}",length=run['length'],system_directory=relative(system),run_record=relative(run_path),source_snapshot=relative(source),clock_objective_mhz=config['hardware']['clock_objective_mhz'],reported_clock_period_ns=period,reported_clock_uncertainties_ns=[float(v) for v in re.findall(r'Clock Uncertainty:\s*([\d.]+)ns',timing)],resources=resources,resource_limits=limits,resources_within_limits=within,timing_met=timing_met,wns_ns=wns,tns_ns=tns,setup_failing_endpoints=setup,whs_ns=whs,hold_failing_endpoints=hold,pulse_width_failing_endpoints=pulse,timing_checks=checks,no_clock_pins=checks['no_clock'],unconstrained_internal_endpoints=checks['unconstrained_internal_endpoints'],drc_errors_or_critical_warnings=errors,bandwidth_feasible=bandwidth,bandwidth_basis='Analytical compiler feasibility only: embedded routed NoC OPTIMAL solution,7 configured positive read/write paths achieved; serialized complete-cost model remains an engineering estimate, no physical throughput/latency observation',noc_paths=[{k:p[k] for k in ('From','To','ReadBW','WriteBW','ReadAchievedBW','WriteAchievedBW','ReadLatency','WriteLatency') if k in p} for p in paths],integrated_eligible=eligible,pdi=relative(pdis[0]),xsa=relative(xsa),reports=[relative(system/name) for name in ('integrated_timing.rpt','integrated_utilization.rpt','integrated_drc.rpt')],physical_board_activity=False)
 for key,pattern,cast in [('source',r'  Source:\s+([^\n]+)',str),('destination',r'  Destination:\s+([^\n]+)',str),('logic_levels',r'Logic Levels:\s+(\d+)',int),('data_delay_ns',r'Data Path Delay:\s+([\d.]+)ns',float)]:
  record.setdefault('critical_path',{})[key]=cast(match(pattern,timing,key)[1].strip())
 return record,solution,traffic

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--system-dir',required=True);ap.add_argument('--implementation-record');ap.add_argument('--output')
 a=ap.parse_args();system=resolve(a.system_dir)
 record,solution,traffic=inspect_system(system,a.implementation_record)
 out=resolve(a.output) if a.output else system/'closure_inspection.json';require(not out.exists(),'Preserve prior closure inspection; choose fresh --output')
 for name,value in [('noc_solution',solution),('noc_traffic',traffic)]:
  dest=out.with_name(out.stem+'_'+name+'.json');require(not dest.exists(),'Preserve prior extracted compiler JSON');dest.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');record[name]=relative(dest)
 out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(output=relative(out),**record),indent=2))
if __name__=='__main__':main()