"""Prepare and time raw-text to packet work on the accessible Windows host.

Local tokenizer/development text only; no model forward, API or device access.
Packets retain complete document windows and sequential batch semantics.
"""
from pathlib import Path
import argparse,copy,json,platform,statistics,struct,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gate_eval import load_config,window_document
HEADER=struct.Struct('<8sIIIIQQQIIII')
MAGIC=b'SENTRY01'

def packet(window, length, document, sample, last, model, threshold):
    ids=list(window['input_ids']);types=list(window['token_type_ids']);mask=list(window['attention_mask'])
    valid=len(ids)
    if not 1<=valid<=length or len(types)!=valid or len(mask)!=valid:raise ValueError('Bad window extent')
    ids += [0]*(length-valid);types += [0]*(length-valid);mask += [0]*(length-valid)
    if any(type(x)is not int or not 0<=x<30522 for x in ids) or any(x not in (0,1) for x in types+mask):raise ValueError('Bad packed input')
    payload=struct.pack('<'+'I'*length,*ids)+struct.pack('<'+'I'*length,*types)+bytes(mask)
    flags=int(last)
    header=HEADER.pack(MAGIC,1,length,valid,window['window_index'],document,sample,model,flags,len(payload),threshold,0)
    assert len(header)==64
    return header+payload

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--length',type=int,choices=[128,256,512],default=256)
    ap.add_argument('--batch',type=int,choices=[1,8,32],default=1)
    ap.add_argument('--condition-note',default='')
    ap.add_argument('--output',type=Path,default=Path('results/goal4/host_preparation/l256_b1'))
    args=ap.parse_args()
    if not args.execute:print(json.dumps(dict(execute=False,length=args.length,batch=args.batch,documents=32,model_forward=False,hardware=False)));return
    cfg=load_config();cfg=copy.deepcopy(cfg);cfg['windowing']['primary_length']=args.length
    out=args.output if args.output.is_absolute() else ROOT/args.output
    if out.exists():raise ValueError('Preserve prior host evidence; choose a new output')
    rows=[json.loads(line) for line in (ROOT/'data/prepared/search_candidate_scoring.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=[]
    for label in (0,1):selected += [row for row in rows if row['label']==label][:16]
    if len(selected)!=32:raise ValueError('Expected 16 benign and 16 attacked development inputs')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/floating/epoch_3',local_files_only=True,use_fast=True)
    prep=json.loads((ROOT/'results/goal4/internal_integration/arena_w8_qat_real_input_v1/preparation.json').read_text())
    threshold=int(prep['threshold_bits_hex'],16);model=0xbf04010000000001
    # Warm tokenizer initialization explicitly outside the timed boundary.
    window_document(tokenizer,selected[0]['text'],selected[0]['document_id'],config=cfg)
    timings=[];payloads=[];records=[];sample=0
    for doc,row in enumerate(selected,1):
        observed=[];packed=None;windows=None
        for repeat in range(3):
            begin=time.perf_counter_ns()
            windows=window_document(tokenizer,row['text'],row['document_id'],config=cfg)
            packed=[packet(w,args.length,doc,sample+i,i==len(windows)-1,model,threshold) for i,w in enumerate(windows)]
            observed.append((time.perf_counter_ns()-begin)/1e6)
        sample+=len(windows);payloads.extend(packed);timings.append(statistics.median(observed))
        records.append(dict(document_id=row['document_id'],label=row['label'],windows=len(windows),raw_utf8_bytes=len(row['text'].encode()),preprocessing_ms_by_repeat=observed,median_preprocessing_ms=timings[-1],packet_bytes=sum(map(len,packed))))
    out.mkdir(parents=True);(out/'workload.bin').write_bytes(b''.join(payloads))
    result=dict(passed=True,condition_note=args.condition_note,torch_imported=('torch' in sys.modules),recorded_host=platform.platform(),python=sys.version,length=args.length,batch=args.batch,
        execution='sequential documents/windows; batch is grouping only, no parallel detector claim',
        boundary='warm raw text -> complete tokenizer windows -> ID/type/mask validation/padding -> little-endian binary packets; excludes tokenizer/model cold load, transport, A72 work and FPGA',
        tokenizer_checkpoint='checkpoints/floating/epoch_3',tokenizer_revision=cfg['models']['detector']['tokenizer_revision'],
        data='data/prepared/search_candidate_scoring.jsonl',documents=32,windows=len(payloads),repeats=3,
        median_document_preprocessing_ms=statistics.median(timings),total_document_preprocessing_ms=sum(timings),
        median_single_window_document_preprocessing_ms=statistics.median([r['median_preprocessing_ms'] for r in records if r['windows']==1]),
        packet_header_bytes=64,response_bytes=64,transport='selected UART8N1; analytical wire time is10 bits per byte; no board observation',
        cold_boundary='tokenizer load and model/arena SD-card load separately charged; not amortized into warm per-window cost',records=records)
    (out/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
