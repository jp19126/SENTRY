"""Run the narrow offline supervisor protocol check against fake peripherals."""
from pathlib import Path
from datetime import datetime, timezone
import json,shutil,subprocess,importlib.util
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/goal4/host_preparation'/('supervisor_protocol_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));out.mkdir(parents=True)
    snapshot=out/'sources';snapshot.mkdir()
    for name in ['sentry_supervisor.c','sentry_supervisor.h']:shutil.copyfile(ROOT/'hardware/host'/name,snapshot/name)
    shutil.copytree(ROOT/'hardware/host/offline',snapshot/'offline')
    command=['gcc','-DSENTRY_HOST_OFFLINE','-std=c11','-O2','-Wall','-Wextra','-Werror','-I'+str(snapshot/'offline'),str(snapshot/'sentry_supervisor.c'),str(snapshot/'offline/supervisor_check.c'),'-o',str(out/'supervisor_check')]
    record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=False,compile_command=command,lengths=[128,256,512],cases_per_length=8,arithmetic_evidence=False,physical_measurement=False,boundary='Same supervisor C with fake UART/SD/cache/MMIO; fake device supplies risks and document maxima; verifies host consumption, validation, identities and error paths only')
    try:
        with (out/'compile.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
        result=subprocess.run([str(out/'supervisor_check'),str(out/'captured_protocol')],text=True,capture_output=True,check=True,timeout=60)
        (out/'check.log').write_text(result.stdout+result.stderr)
        spec=importlib.util.spec_from_file_location('board_runner',ROOT/'scripts/run_board_evaluation.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        parsed=[]
        for length in [128,256,512]:
            data=(out/f'captured_protocol_l{length}.bin').read_bytes();package=dict(length=length,model_id=0xbf04010000000001,threshold_bits=0x3b955a95,timer_frequency_hz=100000000)
            assert runner.ready_response(data[:64],package)==100000000
            maximum=0
            for i in range(3):
                if i==2:maximum=0
                maximum,response=runner.response(data[64+64*i:128+64*i],package['model_id'],10 if i<2 else 11,100+i,1 if i==1 else 0,i>0,maximum,package['threshold_bits'])
                assert response['supervisor_submit_to_completion_ns']==1000
            parsed.append(length)
        record['actual_fake_response_parser_lengths']=parsed
        record['passed']=result.stdout.count('PASS L')==24 and 'HOST_PROTOCOL_CHECK_PASS' in result.stdout
        if not record['passed']:raise RuntimeError('Missing expected host check cases')
    except BaseException as exc:record['error']=str(exc);raise
    finally:(out/'run.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(output=str(out),**record),indent=2))
if __name__=='__main__':main()
