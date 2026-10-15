"""Read actual CNN HLS reports and generate the authoritative host register map."""
import argparse,json,re,xml.etree.ElementTree as ET
from pathlib import Path
from hardware import digest,write_json
def summary(path):
 x=ET.parse(path).getroot()
 def fields(tag):return {e.tag:e.text for e in x.find(tag)}
 return {'estimated_period_ns':float(x.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod')),
  'latency':fields('PerformanceEstimates/SummaryOfOverallLatency'),'resources':{k:int(v) for k,v in fields('AreaEstimates/Resources').items()},'sha256':digest(path)}
def memory(path):
 rows=[]
 for line in Path(path).read_text().splitlines():
  p=[s.strip() for s in line.strip().strip('|').split('|')]
  if len(p)==10 and ('RAM_' in p[1]) and all(v.isdigit() for v in p[2:]):
   rows.append(dict(zip(['name','module','bram18','ff','lut','uram','words','bits','banks','logical_bits'],p[:2]+[int(v) for v in p[2:]])))
 return rows
def report(hls,out):
 hls=Path(hls);out=Path(out);out.mkdir(parents=True,exist_ok=False);reports=hls/'project/solution/syn/report';rtl=hls/'project/solution/syn/verilog/kd_cnn_top_control_s_axi.v'
 modules={n:summary(reports/(n+'_csynth.xml')) for n in ['kd_cnn_top','window','block_r','project','load_model']}
 memories=memory(reports/'kd_cnn_top_csynth.rpt')+memory(reports/'window_csynth.rpt')
 groups={}
 for n,prefix in [('embedding','model_embedding'),('convolution_weights','model_conv_'),('projection_weights','model_projection_'),('activations_a','a_'),('activations_b','b_')]:
  rows=[r for r in memories if r['name'].startswith(prefix)]
  if n=='projection_weights':rows=[r for r in rows if re.fullmatch(r'model_projection_\d+_U',r['name'])]
  if n.startswith('activations_'):rows=[r for r in memories if r['name']==prefix+'U' or re.fullmatch(prefix+r'\d+_U',r['name'])]
  groups[n]={'entries':len(rows),'bram18':sum(r['bram18'] for r in rows),'uram':sum(r['uram'] for r in rows),'logical_bytes':sum(r['logical_bits'] for r in rows)//8,'memory_rows':rows}
  allocated=groups[n]['bram18']*2304+groups[n]['uram']*36864
  groups[n]['nominal_allocated_block_bytes']=allocated;groups[n]['logical_fraction_of_block_capacity']=groups[n]['logical_bytes']/allocated if allocated else None
 params={k:int(v,16) for k,v in re.findall(r"(ADDR_\w+)\s*=\s*\d+'h([0-9a-f]+)",rtl.read_text())}
 actual={'control':params['ADDR_AP_CTRL']}
 for name,signal in [('blob','BLOB'),('bytes','BYTES'),('ids','IDS'),('masks','MASKS'),('count','COUNT'),('load','LOAD'),('first','FIRST'),('final','FINAL_R'),('previous','PREVIOUS'),('previous_windows','PREVIOUS_WINDOWS'),('threshold','THRESHOLD'),('output','OUTPUT_R_R')]:
  actual[name]=params['ADDR_'+signal+'_DATA_0']
  if name in ['blob','ids','masks','output'] and params['ADDR_'+signal+'_DATA_1']!=actual[name]+4:raise ValueError('Noncontiguous pointer register pair')
 header='#ifndef SENTRY_ACTUAL_CNN_REGISTERS\n#define SENTRY_ACTUAL_CNN_REGISTERS\n#include "host_driver.hpp"\nnamespace kdhost { constexpr Registers synthesized_registers={'+','.join(str(v) for v in actual.values())+'}; }\n#endif\n'
 (out/'synthesized_register_map.hpp').write_text(header)
 write_json(out/'hls_summary.json',{'hls':str(hls),'prepared':json.loads((hls/'prepared.json').read_text()),'modules':modules,'memory_groups':groups,'register_map':actual,'control_rtl_sha256':digest(rtl),'boundary':'HLS estimates and inspected generated RTL. No RTL co-sim, routed timing, actual DDR bandwidth or physical measurement inferred.'})
 print(json.dumps({'top':modules['kd_cnn_top'],'memory':{k:{a:b for a,b in v.items() if a!='memory_rows'} for k,v in groups.items()},'register_map':actual}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--hls',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();report(a.hls,a.out)
