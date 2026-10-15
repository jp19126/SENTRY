"""Freeze a common identity contract atomically; never alter an existing contract."""
from common import CANONICAL, SHARED, OUT, lock, read, write, sha

def main():
    base = CANONICAL / 'results/cnn_adoption_v1'
    checkpoint = base / 'runs/bert_matched_lr5e5_supervised_s42_r2/checkpoint_selected.pt'
    expected = '95025bfc0e29163efea613d556d7da0b516ec1bb6c6fac9d9c4c9c6f1f373525'
    assert sha(checkpoint) == expected, 'Shared teacher identity differs'
    with lock('setup'):
        target = SHARED / 'contract.json'
        if target.exists():
            existing = read(target)
            write(OUT/'shared_contract_reuse.json', {'path':str(target), 'sha256':sha(target), 'contract':existing, 'action':'reused unchanged; semantic reconciliation required before execution'})
            print('REUSED', target, sha(target))
            return
        policy = read(CANONICAL/'configs/project.json')['models']['detector']
        tok = CANONICAL/'checkpoints/floating/epoch_3'
        protocol = base/'revision_r2/data_protocol.json'
        p = read(protocol)
        value = {
          'schema':'sentry_dual_track_v1', 'immutable':True, 'created_by':'bert_aie_v1',
          'canonical_root':'/home/jp19126/Projects/SENTRY',
          'shared_root':'/home/jp19126/Projects/SENTRY/results/dual_track_v1/shared',
          'teacher':{'path':str(checkpoint.relative_to(CANONICAL)).replace('\\','/'), 'sha256':expected, 'seed':42, 'selection':'matched-data development-selected epoch4; no confirmation selection', 'backbone':policy},
          'tokenizer':{'path':'checkpoints/floating/epoch_3','files_sha256':{name:sha(tok/name) for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')}},
          'base_data':{'protocol':'results/cnn_adoption_v1/revision_r2/data_protocol.json','protocol_sha256':sha(protocol), 'source_role_membership':p['source_role_membership'], 'payload_role_membership':p['payload_role_membership'], 'raw_role_sha256':p['raw_role_sha256'], 'prepared_root':'results/cnn_adoption_v1/prepared/token_r2', 'exposure':'Existing adoption confirmation exposed; never fresh holdout or training. Fit and development memberships preserved.'},
          'input':{'length':256,'content_length':254,'overlap_content_tokens':64,'stride_content_tokens':190,'complete_tails':True,'special_tokens':'[CLS] content [SEP] then right PAD','token_type_ids':0,'attention_mask':'1 real including specials; 0 PAD','aggregation':'maximum window logit margin','decision':'strict score > threshold','retention':'retain complete raw document until decision'},
          'calibration':{'primary_alpha':0.01,'secondary_alphas':[0.005,0.02,0.05],'procedure':'benign-only conservative order statistic; strict >; no randomized ties','status':'reporting operating points, not deployment requirements'},
          'new_data':{'manifest':'prospective_v1/manifest.json','source':'unused permitted inventory groups excluding every old role; new authored payload ancestry, construction limits explicit','release':'Each track freezes candidates before scoring; confirmation score files remain task-private until both candidate freezes'},
          'timing':{'manifest':'prospective_v1/timing_manifest.json','boundary':'raw document through tokenizer, all windows/tails, transfers, inference, return, aggregation, decision','batch_windows':1,'cold_warm':'separate model loading/graph creation from resident warm checks','quantiles':[0.5,0.9,0.95]},
          'target':{'board':'VEK280','part':'xcve2802-vsvh1760-2MP-e-S','physical_permission':False},
          'resource_locks':{'protocol':'atomic mkdir on canonical shared path; owner.json; never steal a lock; release on exit','setup':'locks/setup.lock','gpu0':'locks/gpu0.lock','vendor':'locks/vendor.lock'},
          'closed':['project final-calibration','project final-test','NotInject including redistributions'],
          'limits':'No paid service, external model API, credential access or physical device action.'
        }
        write(target,value)
        write(OUT/'shared_contract_reuse.json', {'path':str(target),'sha256':sha(target),'action':'created before any new score or calibration'})
        print('CREATED',target,sha(target))

if __name__ == '__main__': main()
