"""Source-derived whole-window counts, explicitly separate from measured traffic."""
from pathlib import Path
import re
from common import ROOT, OUT, read, sha, write


def main():
    source=ROOT/'hardware/bert_aie_v1/model.hpp'
    text=source.read_text(encoding='utf-8')
    def constant(name):
        match=re.search(r'\b'+name+r'=(\d+)',text)
        if not match:raise ValueError('Missing literal dimension '+name)
        return int(match.group(1))
    n,d,f,h,layers=[constant(k) for k in ('N','D','F','HEAD','LAYERS')]
    wm,wk,wn,am,ak,an=[constant(k) for k in ('WM','WK','WN','AM','AK','AN')]
    heads=d//h
    assert n%d==0 and d%h==0 and n%wm==0 and d%wk==0 and f%wk==0
    linears=[('q',d,d),('k',d,d),('v',d,d),('o',d,d),('ff1',d,f),('ff2',f,d)]
    dense_calls=layers*sum((n//wm)*(k//wk)*(o//wn) for _,k,o in linears)
    qk_calls=layers*heads*(n//am)*(n//an)*(h//ak)
    pv_calls=layers*heads*(n//am)*(h//an)*(n//ak)
    dense_payload={'a':dense_calls*wm*wk,'b':dense_calls*wk*wn,'c':dense_calls*wm*wn*4}
    attention_payload={'a':(qk_calls+pv_calls)*am*ak*2,'b':(qk_calls+pv_calls)*ak*an*2,'c':(qk_calls+pv_calls)*am*an*4}
    hidden=n*d
    # Logical accesses to full intermediate storage as written in model.hpp.
    # Per layer, q/k/v attention panels reread each input n/an or n/am times.
    att_reads=heads*((n//am)*(n//an)*am*h+(n//am)*(n//an)*h*an+(n//am)*(h//an)*(n//ak)*ak*an)
    layer_reads=sum(n*k for _,k,_ in linears)+att_reads+2*hidden+3*hidden+n*f+2*hidden+3*hidden
    layer_writes=sum(n*o for _,_,o in linears)+hidden+hidden+hidden+n*f+hidden+hidden
    scratch_read=3*hidden+layers*layer_reads+d*d
    scratch_write=2*hidden+layers*layer_writes
    bytes_per_port_second=16*250_000_000
    manifest=OUT/'export/w8a8_bf16_attention/manifest.json'
    result={
        'schema':'sentry_bert_aie_v1_source_costs_v1',
        'source_sha256':{'model.hpp':sha(source),'pl.cpp':sha(ROOT/'hardware/bert_aie_v1/pl.cpp'),'tiles.cpp':sha(ROOT/'hardware/bert_aie_v1/tiles.cpp')},
        'export_manifest_sha256':sha(manifest),
        'dimensions':{'sequence':n,'hidden':d,'ff':f,'heads':heads,'head_width':h,'layers':layers},
        'matrix_macs':{'encoder_dense':layers*sum(n*k*o for _,k,o in linears),'encoder_attention':layers*2*n*n*d,'pooler_classifier':d*d+2*d},
        'engine_calls':{'int8':dense_calls,'bf16_qk':qk_calls,'bf16_pv':pv_calls},
        'plio_payload_bytes':{'int8':dense_payload,'bf16':attention_payload,'total':sum(dense_payload.values())+sum(attention_payload.values())},
        'double_buffer_data_bytes':{'int8':2*(wm*wk+wk*wn+wm*wn*4),'bf16':2*(am*ak*2+ak*an*2+am*an*4)},
        'resident_storage_bytes':{'full_fp32_intermediates':(7*hidden+n*f)*4,'quantized_activation':n*f,'excludes':'temporary operand/result tiles, score panels, norm buffers, stream FIFOs and compiler replication'},
        'external_scratch_logical_accesses':{'reads_float32':scratch_read,'writes_float32':scratch_write,'bytes':4*(scratch_read+scratch_write),'per_layer_reads':layer_reads,'per_layer_writes':layer_writes,
            'boundary':'Source-level scalar tensor accesses; repeated same-address expression loads assumed CSE. Not measured AXI traffic: caching, coalescing, burst widening and compiler scheduling can change physical bytes. Resident mapping relocates these accesses rather than eliminating their compute/local-storage cost.'},
        'other_logical_payloads':{'input_int32_bytes':3*n*4,'output_logits_status_bytes':3*4,'embedding_value_reads_bytes':3*hidden*4,'dense_weight_reads_bytes':layers*(n//wm)*sum(k*o for _,k,o in linears),
            'excludes':'normalization/head/scale/bias parameter accesses, AXI transaction overhead, padding, repetition/cache behavior and initial model transfer'},
        'port_bound':{'assumed_pl_clock_hz':250_000_000,'bits':128,'ideal_bytes_per_second_per_port':bytes_per_port_second,'int8_a_seconds':dense_payload['a']/bytes_per_port_second,
            'boundary':'Ideal necessary bound for this graph at the requested port clock. Achieved routed clock and sustained bandwidth are unmeasured; not a general BERT/FPGA bound or prediction of complete latency.'},
        'pl_only_comparison':'Buffered PL has the same operation counts and can hold operator tiles, but dense multiplier allocation/clock and FP32 nonlinear costs are not measured here. No PL-only advantage claim; the implemented external/resident heterogeneous pair isolates scratch placement.',
        'evidence_type':'Analytical source counts; no synthesis, simulation-cycle, routed timing, board power or energy claim.'}
    assert result['matrix_macs']['encoder_dense']==805306368 and result['matrix_macs']['encoder_attention']==134217728
    assert dense_calls==3072 and qk_calls+pv_calls==2048
    write(OUT/'source_cost_model.json',result)
    print({'path':str(OUT/'source_cost_model.json'),'macs':result['matrix_macs'],'calls':result['engine_calls'],'plio_bytes':result['plio_payload_bytes']['total'],'scratch_logical_bytes':4*(scratch_read+scratch_write)})


if __name__=='__main__':main()
