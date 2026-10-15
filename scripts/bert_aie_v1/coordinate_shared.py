"""Score-free reconciliation with concurrently running CNN track via shared files."""
from common import CANONICAL, SHARED, OUT, lock, read, write, sha

def main():
    ready=read(SHARED/'kd_cnn_v2_data_ready.json')
    assert ready['common_contract_sha256']==sha(SHARED/'contract.json')
    assert ready['status']=='frozen_prepared_score_free'
    protocol=read(SHARED/ready['data_protocol'])
    assert protocol['frozen_before_new_model_scoring'] and not protocol['protected_data_read']
    assert sha(SHARED/ready['data_protocol'])==ready['data_protocol_sha256']
    assert sha(SHARED/ready['prepared_summary'])==ready['prepared_summary_sha256']
    with lock('setup'):
        record={'task':'bert_aie_v1','kind':'score_free_interface_and_resource_coordination',
            'common_contract_sha256':sha(SHARED/'contract.json'),
            'data_decision':'Adopt sibling-provided frozen data/prepared/new_calibration and new_confirmation as the primary common panel BEFORE any BERT development or confirmation scoring. BERT continues existing base development for format qualification. Own prospective_v1 panel is preserved unused; not used to choose model/format/threshold.',
            'primary_protocol':ready['data_protocol'],'primary_protocol_sha256':ready['data_protocol_sha256'],
            'prepared_summary_sha256':ready['prepared_summary_sha256'],
            'roles':{'calibration':'data/prepared/new_calibration','selection':'data/prepared/new_confirmation'},
            'timing':'Please reuse prospective_v1/timing_manifest.json, development-only, exact IDs,20repeats,3warmups, full raw text to decision.',
            'gpu_request':'BERT requests a short exclusive window (~4 corpus development scorings plus graph/eager profiles; estimated under5min after cache release). All jobs acquire gpu0.lock and honor owner, no stealing.',
            'vendor_request':'Our initial AIE compile ended; idle vendor lock released. CNN may run initial HLS next. AIE retry needs low-memory-contention window (initial compile used ~12GiB and briefly reduced Windows free below2GiB), please do not overlap GPU training with AIE compile. We will wait while gpu0 held.',
            'results_release':'BERT confirmation scores remain task-private until both candidate freezes. No sibling confirmation outcomes inspected.',
            'supersession':'Immutable contract remains byte-identical. This score-free data-path supplement reconciles concurrently prepared material.'}
        write(SHARED/'bert_aie_v1_coordination.json',record)
        write(OUT/'shared_data_selection.json',record)
        config_path=CANONICAL/'configs/bert_aie_v1.json';config=read(config_path)
        config['primary_confirmation_data']={'shared_protocol_sha256':ready['data_protocol_sha256'],**record['roles'], 'selection_before_any_new_scoring':True}
        config['preserved_unused_panel']='results/dual_track_v1/shared/prospective_v1; only timing manifest used'
        write(config_path,config)
        print(record)

if __name__=='__main__':main()
