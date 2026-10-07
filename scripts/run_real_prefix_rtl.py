"""Six saved-input RTL calls: embedding add -> LayerNorm -> layer0-query A8.

The unchanged testbench's two slots carry different128-row halves, not repeated
measurements. No model forward, synthesis, compilation or dataset access.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, math, re, shutil, struct
import run_fixed_fp32_rtl as rtl
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/goal4/numerical_bridge'
REFERENCE=DATA/'ordered_model_v1'
OUT=DATA/'rtl_prefix_v1'
CASE='aeslc:train:panus-s_inbox_1.subject:clean'
REVISION='panel_cache_batched_attention_v1'
SHAPE=dict(rows=128,width=256,outputs=0,bias=False)

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def need(condition,message):
    if not condition:raise ValueError(message)
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def floats(raw):return struct.unpack('<'+'f'*(len(raw)//4),raw)
def load(name,length,base=DATA):
    raw=(base/name).read_bytes();need(len(raw)==length,'Extent mismatch: '+name);return raw

def compare_float(actual,expected,atol,rtol):
    av,ev=floats(actual),floats(expected)
    ai=struct.unpack('<'+'I'*len(av),actual);ei=struct.unpack('<'+'I'*len(ev),expected)
    differences=[abs(float(a)-float(e)) for a,e in zip(av,ev)]
    bad=[i for i,(a,e,d) in enumerate(zip(av,ev,differences)) if not math.isfinite(a) or not math.isfinite(e) or d>atol+rtol*abs(e)]
    return dict(values=len(av),passed=not bad,atol=atol,rtol=rtol,
                bit_differences=sum(a!=e for a,e in zip(ai,ei)),
                maximum_absolute_error=max((d for d in differences if math.isfinite(d)),default=None),
                bound_failures=len(bad),first_failure_indices=bad[:8])

def compare_codes(actual,expected):
    need(len(actual)==len(expected),'Code reference size differs')
    bad=[i for i,(a,b) in enumerate(zip(actual,expected)) if a!=b]
    return dict(values=len(actual),passed=not bad,exact_code_differences=len(bad),first_failure_indices=bad[:8])

def report(record):
    lines=['# Saved-input real-RTL prefix', '',
      'This diagnostic chains only embedding addition, embedding LayerNorm and layer0-query A8 quantization for the existing saved L256 input. Actual RTL output bytes feed the next operation. It does not execute the whole detector or establish general numerical/protection acceptance.', '',
      'The unchanged compiled six-port service testbench supplies two slots per invocation. Slot0 is tokens0..127; slot1 is tokens128..255. These are six distinct required calls across three invocations, not repeated measurements.', '',
      'References: embedding uses explicit separate FP32 additions; LayerNorm uses saved CPU ordered model_embedding_ln with the existing local absolute/relative bounds2e-4/2e-4; A8 requires exact saved model_layer0_query_a8 and exact same-input nearest-even/clip codes from actual RTL LayerNorm outputs. Floating bit differences are reported separately. No full-model tolerance is introduced.', '',
      '| Stage | State | Cycles per half | Observed comparison |','|---|---|---|---|']
    for stage in record.get('stages',[]):
        lines.append('|'+stage['name']+'|'+stage['state']+'|'+str(stage.get('cycles_per_half'))+'|'+json.dumps(stage.get('comparison',{}),separators=(',',':'))+'|')
    lines += ['', 'State: '+record['state']+'. '+record.get('error',''), '',
      'Source/capture associations, exact commands, raw inputs/outputs, logs and failure records are preserved under results/goal4/numerical_bridge/rtl_prefix_v1. The simulator runs a private copy of the already compiled snapshot; no RTL, testbench, compilation or synthesis changes are made.', '',
      'Cycles retain the existing accepted-service-start to sampled-done convention and bounded six-port memory model. Their sum covers this arithmetic prefix only. Python orchestration/file handoffs are outside simulated cycles; this is not end-to-end execution or board timing.', '',
      f'Reproduction: `wsl.exe -d Ubuntu --exec /usr/bin/python3 {ROOT.as_posix()}/scripts/run_real_prefix_rtl.py --execute`. Default is preview; completed/failed output is never overwritten. The existing pause marker is respected.']
    (ROOT/'reports/goal4_real_prefix_rtl.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    if not args.execute:
        print(json.dumps(dict(execute=False,case=CASE,operations=['embedding_add','embedding_ln','layer0_query_a8'],calls=6,new_synthesis=False,model_forward=False)));return
    rtl.paused();need(not OUT.exists(),'Preserve existing prefix evidence; no overwrite')
    config=read(ROOT/'configs/project.json');source,synthesis=rtl.source_evidence(config)
    manifest=read(DATA/'manifest.json');reference=read(REFERENCE/'summary.json')
    need(manifest['case']==reference['case']==CASE and reference['completed'] and reference['validation_device']=='cpu','Saved CPU reference association differs')
    need(manifest['hardware_source_revision']==reference['service_revision']==synthesis['service_schedule_revision']==REVISION,'Service revision differs')
    need(reference['saved_ordered_reference_reproduced_exactly'] and reference['validation_shape']==[1,256] and all(v==8 for v in reference['precision_map'].values()),'Saved ordered W8 reference differs')
    need(manifest['checkpoint'].replace('\\','/')==reference['checkpoint'].replace('\\','/')=='checkpoints/quantized/qat_w8_a8','Checkpoint association differs')
    tag=re.sub(r'[^A-Za-z0-9_-]','_',synthesis['recorded_utc']);build=ROOT/'build/fixed_fp32_rtl'/tag
    compiled=read(build/'compiled.json')
    need(compiled['signature']==dict(synthesis_recorded_utc=synthesis['recorded_utc'],service_schedule_revision=REVISION),'Compiled signature differs')
    for name in ('fixed_fp32_axi_tb.sv','linear_axi_tb.sv'):
        need((build/name).read_bytes()==(ROOT/'hardware'/name).read_bytes(),'Compiled testbench differs: '+name)
    for name in compiled['reachable_rtl_files']:
        need((build/'rtl'/name).read_bytes()==(source/'project/solution/syn/verilog'/name).read_bytes(),'Compiled RTL copy differs: '+name)
    original=build/'ip_project/fixed_fp32_sim.sim/sim_1/behav/xsim';snapshot=compiled['snapshot']
    need((original/'xsim.dir'/snapshot/'xsimk').is_file(),'Compiled executable missing')
    OUT.mkdir();sim=OUT/'sim';sim.mkdir()
    shutil.copytree(original/'xsim.dir',sim/'xsim.dir',ignore=shutil.ignore_patterns('xsim_script.tcl','xsimSettings.ini','TempBreakPointFile.txt','*.log','*.jou','*.wdb'))
    for path in [*original.glob('*.dat'),original/'xsim.ini']:shutil.copyfile(path,sim/path.name)
    (sim/'run.tcl').write_text('run all\nquit\n')
    for path in [Path(__file__),Path(rtl.__file__),build/'compiled.json',source/'run.json',DATA/'manifest.json',REFERENCE/'summary.json']:
        destination=OUT/(('synthesis_' if path==source/'run.json' else 'reference_' if path==REFERENCE/'summary.json' else '')+path.name)
        shutil.copyfile(path,destination)
    for name in ('fixed_fp32_axi_tb.sv','linear_axi_tb.sv','fixed_fp32_service.hpp','fixed_fp32_service_tb.cpp'):
        shutil.copyfile(ROOT/'hardware'/name,OUT/name)
    record=dict(state='preparing',passed=False,case=CASE,recorded_utc=datetime.now(timezone.utc).isoformat(),
       service_revision=REVISION,synthesis_recorded_utc=synthesis['recorded_utc'],compiled_source=str(build),
       source_association_checked=True,checkpoint=reference['checkpoint'],reference_directory=str(REFERENCE),
       no_new_compilation=True,no_new_synthesis=True,clock_ns=synthesis['clock_ns'],memory_model=rtl.MEMORY_MODEL,
       slot_semantics='slot0 tokens0..127; slot1 tokens128..255; not experimental repetitions',
       stages=[],full_detector_acceptance=None)
    def save():rtl.save(OUT/'run.json',record);report(record)
    save()
    xsim=str(Path(config['hardware']['hls']['vivado_executable']).parent/'xsim')
    def execute(name,op,arrays,scale=1.0):
        rtl.paused();need(read(source/'run.json')==synthesis,'Synthesis changed during prefix')
        folder=OUT/name;folder.mkdir()
        case=dict(SHAPE,op=op,scale=scale,id=name)
        extents=None
        for slot,values in enumerate(arrays):
            current={}
            for port in ('x','y','z','accumulators'):
                raw=values.get(port,b'\0'*4);current[port]=len(raw)
                (folder/f'{port}_{slot}.hex').write_text(''.join(f'{v:02x}\n' for v in raw))
            current.update(output=4 if op==1 else 32768*4,codes=32768 if op==1 else 4)
            need(extents is None or extents==current,'Half extents differ');extents=current
        values=dict(OP=op,ROWS=128,WIDTH=256,OUTPUTS=0,BIAS=0,SCALE=struct.pack('>f',scale).hex(),COUNT=32768,CASE_DIR=str(folder),CLOCK_NS=synthesis['clock_ns'],TIMEOUT_CYCLES=10000000)
        values.update({k.upper()+'_BYTES':v for k,v in extents.items()})
        command=[xsim,snapshot,'-tclbatch',str(sim/'run.tcl')]
        for k,v in values.items():command += ['-testplusarg',str(k)+'='+str(v)]
        stage=dict(name=name,state='running',op=op,command=command,extents=extents,token_ranges=[[0,128],[128,256]])
        record['stages'].append(stage);record['state']='running';save()
        rc=rtl.command_run(command,sim,folder/'simulation.log',timeout=900)
        tx,traffic,captured,issues=rtl.parse_capture((folder/'simulation.log').read_text(errors='replace'),case,synthesis['clock_ns'])
        stage.update(returncode=rc,transactions=tx,traffic=traffic,captured=captured,issues=issues,
                     cycles_per_half=[v['latency_cycles'] for v in tx])
        save();need(rc==0 and not issues,'RTL capture/status failed: '+name)
        parts=[]
        for slot in range(2):
            tokens=(folder/f'actual_{slot}.hex').read_text().split()
            need(all(re.fullmatch('[0-9a-fA-F]{2}',v) for v in tokens),'Unknown output bytes')
            raw=bytes(int(v,16) for v in tokens);need(len(raw)==32768*(1 if op==1 else 4),'Output extent differs')
            parts.append(raw)
        actual=b''.join(parts);(folder/'actual_joined.bin').write_bytes(actual)
        return actual,stage
    try:
        word,type_,position=[load('embedding_'+name+'.bin',262144) for name in ('word','type','position')]
        needed=[floats(raw) for raw in (word,type_,position)]
        expected=struct.pack('<'+'f'*65536,*[f32(f32(a+b)+c) for a,b,c in zip(*needed)])
        actual,stage=execute('embedding_add',3,[dict(x=word[i:i+131072],y=type_[i:i+131072],z=position[i:i+131072]) for i in (0,131072)])
        (OUT/'embedding_add/expected_separate_fp32.bin').write_bytes(expected)
        stage['comparison']=compare_float(actual,expected,0.0,0.0);stage['state']='passed' if stage['comparison']['passed'] else 'failed';save()
        need(stage['comparison']['passed'],'Embedding separate-FP32 sum mismatch')
        params=load('embedding_ln.bin',2048)
        actual,stage=execute('embedding_ln',5,[dict(x=actual[i:i+131072],y=params[:1024],z=params[1024:]) for i in (0,131072)])
        expected=load('model_embedding_ln.bin',262144,REFERENCE)
        (OUT/'embedding_ln/expected_ordered_cpu.bin').write_bytes(expected)
        stage['comparison']=compare_float(actual,expected,2e-4,2e-4);stage['state']='passed' if stage['comparison']['passed'] else 'failed';save()
        need(stage['comparison']['passed'],'LayerNorm existing local bound exceeded')
        scale=struct.unpack('<f',load('layer0_query_params.bin',2052)[:4])[0];need(math.isfinite(scale) and scale>0,'Invalid saved A8 scale')
        codes=[]
        for value in floats(actual):
            quotient=f32(value/scale);need(math.isfinite(quotient),'Nonfinite prefix A8 quotient')
            codes.append(max(-128,min(127,round(quotient))))
        same_input=struct.pack('<'+'b'*65536,*codes)
        actual,stage=execute('layer0_query_a8',1,[dict(x=actual[i:i+131072]) for i in (0,131072)],scale)
        expected=load('model_layer0_query_a8.bin',65536,REFERENCE)
        (OUT/'layer0_query_a8/expected_ordered_a8.bin').write_bytes(expected)
        (OUT/'layer0_query_a8/expected_same_input_a8.bin').write_bytes(same_input)
        comparison=dict(saved_ordered_codes=compare_codes(actual,expected),same_rtl_ln_input=compare_codes(actual,same_input))
        comparison['passed']=all(comparison[k]['passed'] for k in ('saved_ordered_codes','same_rtl_ln_input'))
        stage['comparison']=comparison;stage['state']='passed' if comparison['passed'] else 'failed';save()
        need(comparison['passed'],'Discrete A8 mismatch; stop chosen prefix')
        record.update(state='returned',passed=True,executed_calls=6,arithmetic_prefix_cycles=sum(sum(v['cycles_per_half']) for v in record['stages']))
        save();print(json.dumps(dict(passed=True,evidence=str(OUT/'run.json'),stages=[{k:v[k] for k in ('name','state','cycles_per_half','comparison')} for v in record['stages']]),indent=2))
    except Exception as exc:
        record.update(state='stopped',passed=False,error=str(exc));save();raise
if __name__=='__main__':main()
