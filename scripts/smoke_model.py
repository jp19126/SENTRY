"""Goal 1 only: load pretrained BERT-Mini and score real development emails."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from gate_eval import load_config, score_documents
    config = load_config()
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no GPU validation is claimed.')
        torch.cuda.manual_seed_all(seed)
    model_config = config['models']['detector']
    if not model_config['revision'] or not model_config['tokenizer_revision']:
        raise ValueError('Pin actual source/tokenizer revisions in configs/project.json first.')
    cache = ROOT / config['paths']['checkpoints'] / 'huggingface'
    tokenizer = AutoTokenizer.from_pretrained(model_config['id'], revision=model_config['tokenizer_revision'], cache_dir=cache, use_fast=True, token=False)
    model, loading = AutoModelForSequenceClassification.from_pretrained(model_config['id'], revision=model_config['revision'], cache_dir=cache, num_labels=2, output_loading_info=True, token=False)
    assert (model.config.num_hidden_layers, model.config.hidden_size, model.config.num_attention_heads) == (4, 256, 4)
    assert all(k.startswith('classifier.') for k in loading['missing_keys']), loading
    assert not loading['mismatched_keys'] and not loading['error_msgs'], loading
    model.to(args.device).eval()
    pair_path = ROOT / config['paths']['data'] / 'prepared' / 'development_pairs.jsonl'
    pairs = [json.loads(line) for line in pair_path.read_text(encoding='utf-8').splitlines() if line]
    # Spread the bring-up over real source groups; never inspect final test documents.
    selected = []
    seen = set()
    for pair in pairs:
        if pair['source_group_id'] not in seen:
            selected.append(pair)
            seen.add(pair['source_group_id'])
        if len(selected) == 5:
            break
    documents = [dict(pair[side]) for pair in selected for side in ('clean', 'attacked')]
    longest = max((pair['attacked'] for pair in pairs), key=lambda d: len(d['text']))
    if longest['document_id'] not in {d['document_id'] for d in documents}:
        documents.append(dict(longest))
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    start = time.perf_counter()
    scored = score_documents(model, tokenizer, documents, batch_size=args.batch_size, device=args.device, config=config)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # The real tokenizer/window path must preserve every supplied injected span.
    span_checks = []
    for doc in documents:
        windows = [w for w in scored['windows'] if w['document_id'] == doc['document_id']]
        if doc.get('injection_spans'):
            assert any(w.get('window_label') == 1 for w in windows), doc['document_id']
            span_checks.append(doc['document_id'])
    result = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'scope':'Goal 1 model and window bring-up only', 'trained_detector':False, 'classification_head':'randomly initialized, not evidence of detection quality', 'command':f'python scripts/smoke_model.py --device {args.device} --batch-size {args.batch_size}', 'config':config, 'seed':seed, 'python':sys.version, 'torch':torch.__version__, 'transformers':transformers.__version__, 'cuda_runtime':torch.version.cuda, 'device':args.device, 'device_name':torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU', 'parameter_count':sum(p.numel() for p in model.parameters()), 'loading_info':loading, 'source':str(pair_path.relative_to(ROOT)), 'selected_pair_ids':[p['pair_id'] for p in selected], 'injected_documents_with_positive_windows':span_checks, 'elapsed_bringup_seconds':elapsed, 'timing_note':'Includes tokenization and first inference; not a warm steady-state benchmark.', 'document_count':len(scored['documents']), 'window_count':len(scored['windows']), 'long_document_count':sum(d['is_long_document'] for d in scored['documents'])}
    destination = ROOT / config['paths']['results'] / 'goal1'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'smoke_model.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for name in ('windows','documents'):
        (destination / f'smoke_{name}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in scored[name]), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('trained_detector','device_name','parameter_count','document_count','window_count','long_document_count','elapsed_bringup_seconds')},indent=2))
    print(f'Saved model bring-up evidence under {destination}')


if __name__ == '__main__':
    main()
