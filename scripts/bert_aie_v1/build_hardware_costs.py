"""Extract existing AIE build/simulator facts; never runs a simulator/device."""
from pathlib import Path
import hashlib,json,re,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
BUILD=ROOT/'build/bert_aie_v1/common'
REPORT=BUILD/'Work/reports'
mapping=(REPORT/'graph_mapping_analysis_report.txt').read_text()
allocations={}
for bank,name,address,size in re.findall(r'^(MG\(\d+,\d+\):\d+)\s+(\w+)\s+(0x[0-9a-f]+)\s+(\d+)\s',mapping,re.M):
    allocations[name]={'memory_group_bank':bank,'address':address,'bytes':int(size)}
tree=ET.parse(REPORT/'graph.xpe')
module=tree.find('.//AIE_MODULE')
tiles=[dict(tile.attrib) for tile in module.findall('TILE')]
def memory_usage(name,kind):
    return {core:{'used':int(used),'allotted':int(allotted)} for core,used,allotted in
        re.findall(r'Core (\d+_\d+)\s+'+kind+r' Size Used = (\d+)\s+'+kind+r' Size Allotted = (\d+)',
                   (REPORT/name).read_text())}
out={
    'boundary':'Compiler allocation and two-iteration native AIE simulation facts. No whole-model latency or physical-device extrapolation.',
    'compute_tiles':[[10,0],[11,0]],
    'compiler_xpe_tiles':tiles,
    'compiler_xpe_tile_count':int(module.attrib['num_tiles']),
    'aie_configured_clock_mhz':float(module.attrib['clk_freq']),
    'plio_requested_clock_mhz':250,
    'routed_system_clock_mhz':None,
    'allocations':allocations,
    'allocated_graph_buffer_bytes':sum(v['bytes'] for k,v in allocations.items() if k.startswith('buf')),
    'allocated_system_bytes':sum(v['bytes'] for k,v in allocations.items() if k.startswith('sysmem')),
    'distinct_reported_memory_group_banks':sorted({v['memory_group_bank'] for v in allocations.values()}),
    'compiler_xpe_mem_banks_sum':sum(int(t['mem_banks']) for t in tiles),
    'program_memory_bytes':memory_usage('report_pm.txt','PM'),
    'stack_bytes':memory_usage('report_stack.txt','Stack'),
    'stall_profile':None,
    'stall_profile_reason':'No profile/VCD run requested; no stall counters inferred from payload timestamps.',
    'output_timestamps':{},
    'provenance':{}
}
for kind in ('i8','bf'):
    path=BUILD/'aiesimulator_output/data'/f'{kind}_c.txt'
    timestamps=[float(x)*{'ns':1,'us':1000,'ms':1e6,'s':1e9}[unit]
                for x,unit in re.findall(r'^T\s+([\d.eE+-]+)\s+(ns|us|ms|s)',path.read_text(),re.M)]
    assert len(timestamps)==512,(kind,len(timestamps))
    iterations=[{'first_beat_ns':timestamps[i*256],'last_beat_ns':timestamps[i*256+255],
                 'first_to_last_span_ns':timestamps[i*256+255]-timestamps[i*256]}
                for i in range(2)]
    out['output_timestamps'][kind]={'iterations':iterations,'beats_per_iteration':256,
        'bytes_per_beat':16,'first_beat_interval_ns':timestamps[256]-timestamps[0],
        'boundary':'Two independent input tiles supplied by simulator PLIO sources; not connected PL scheduling or steady-state model service time.'}
for path in [REPORT/'graph_mapping_analysis_report.txt',REPORT/'graph.xpe',REPORT/'report_pm.txt',REPORT/'report_stack.txt',
             BUILD/'aiesimulator_output/data/i8_c.txt',BUILD/'aiesimulator_output/data/bf_c.txt',BUILD/'aiesim_validation.json']:
    out['provenance'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
target=BUILD/'native_costs.json'
target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('allocated_graph_buffer_bytes','allocated_system_bytes','compiler_xpe_tile_count','output_timestamps')}))
