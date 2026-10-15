"""Read completed HLS XML/logs; estimates are not connected runtime or routed timing."""
from pathlib import Path
import argparse,hashlib,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser();ap.add_argument('mapping',choices=['resident','external']);ap.add_argument('--attempt',type=int,default=0);args=ap.parse_args()
base=ROOT/'build/bert_aie_v1'/args.mapping
reports=base/'hls/hls/syn/report'
modules={};provenance={}
def unpack(node):
    return {c.tag:unpack(c) for c in node} if len(node) else node.text
for path in reports.glob('*_csynth.xml'):
    x=ET.parse(path).getroot();name=x.findtext('UserAssignments/TopModelName')
    def section(key):
        node=x.find(key)
        return {c.tag:c.text for c in node} if node is not None else {}
    loops=x.find('PerformanceEstimates/SummaryOfLoopLatency')
    modules[name]={'latency':section('PerformanceEstimates/SummaryOfOverallLatency'),
        'estimated_clock_ns':x.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod'),
        'resources':section('AreaEstimates/Resources'),
        'loops':{loop.tag:unpack(loop) for loop in loops.iter() if loop.tag.startswith('VITIS_LOOP_')} if loops is not None else {}}
    provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
x=ET.parse(reports/'bert_pl_csynth.xml').getroot()
target=float(x.findtext('UserAssignments/TargetClockPeriod'));uncertainty=float(x.findtext('UserAssignments/ClockUncertainty'))
estimated=float(x.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod'))
ports={p.findtext('name'):int(p.findtext('Bits')) for p in x.findall('InterfaceSummary/RtlPorts')
       if p.findtext('name').endswith(('_RDATA','_WDATA','_TDATA'))}
out={'boundary':'HLS estimates and scheduling facts; not physical timing or measured connected runtime. Module latency ranges may independently combine K/O limits that no actual encoder call uses.',
     'mapping':args.mapping,'attempt':args.attempt,'clock':{'target_ns':target,'uncertainty_ns':uncertainty,'effective_budget_ns':target-uncertainty,
     'estimated_ns':estimated,'effective_margin_ns':target-uncertainty-estimated,'routed_timing_met':None},
     'interface_bits':ports,'modules':modules,'provenance':provenance,
     'warnings':[line for line in (base/f'hls_attempt{args.attempt}.log').read_text().splitlines() if line.startswith('WARNING:')]}
linear=modules['linear_AieTiles_s'];quant=modules['linear_AieTiles_Pipeline_VITIS_LOOP_73_1']['latency']
linear_min=int(linear['latency']['Best-caseLatency']);linear_max=int(linear['latency']['Worst-caseLatency'])
panel=int(linear['loops']['VITIS_LOOP_76_4']['IterationLatency'])
tile_min=int(linear['loops']['VITIS_LOOP_74_3']['IterationLatency']['range']['min'])
row_min=int(linear['loops']['VITIS_LOOP_74_2']['IterationLatency']['range']['min'])
qmin=int(quant['Best-caseLatency']);qmax=int(quant['Worst-caseLatency'])
row_extra=row_min-8*tile_min;tile_extra=tile_min-panel
linear_extra=linear_min-qmin-8*row_min
def shaped(K,O):
    q=qmin+(K//256-1)*(qmax-qmin)//3
    return linear_extra+q+8*(row_extra+(O//32)*(tile_extra+(K//256)*panel))
panel_trip_max=int(linear['loops']['VITIS_LOOP_76_4']['TripCount']['range']['max'])
reconstructed_generic_max=linear_extra+qmax+8*(row_extra+32*(tile_extra+panel_trip_max*panel))
assert shaped(256,256)==linear_min and reconstructed_generic_max==linear_max
model=modules['model_'+('true' if args.mapping=='resident' else 'false')+'_AieTiles_s']
layer_bounds=model['loops']['VITIS_LOOP_136_3']['IterationLatency']['range']
fixed_layer=int(layer_bounds['min'])-6*linear_min
assert fixed_layer==int(layer_bounds['max'])-6*linear_max
calls={'K256_O256':shaped(256,256),'K256_O1024':shaped(256,1024),'K1024_O256':shaped(1024,256)}
layer_actual=fixed_layer+4*calls['K256_O256']+calls['K256_O1024']+calls['K1024_O256']
nonlayer=int(modules['bert_pl']['latency']['Worst-caseLatency'])-4*int(layer_bounds['max'])
cycles=nonlayer+4*layer_actual
out['actual_shape_schedule_estimate']={'linear_call_cycles':calls,'fixed_layer_cycles':fixed_layer,
    'actual_layer_cycles':layer_actual,'conservative_nonlayer_cycles':nonlayer,'whole_cycles':cycles,
    'seconds_at_requested_clock':cycles*target*1e-9,'formula_constants':{'quant_min':qmin,'quant_max':qmax,
    'k_panel':panel,'tile_extra':tile_extra,'row_extra':row_extra,'linear_extra':linear_extra,'generic_k_panel_trip_max':panel_trip_max},
    'assumptions':'Reconstructs reported loop schedule at actual K/O call shapes; conservative remaining overhead. AXI/AIE streams assumed ready by HLS latency estimate; actual stalls, routed clock and whole-model runtime unmeasured.',
    'anchors':['linear_AieTiles_s: VITIS_LOOP_74_2,74_3,76_4','linear_AieTiles_Pipeline_VITIS_LOOP_73_1','model_*_AieTiles_s: VITIS_LOOP_136_3']}
(base/'hls_costs.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'clock':out['clock'],'interface_bits':ports,'top':modules['bert_pl'],
                 'actual_shape_schedule_estimate':out['actual_shape_schedule_estimate'],'warning_count':len(out['warnings'])}))
