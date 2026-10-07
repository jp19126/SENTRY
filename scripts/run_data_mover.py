"""Focused simulation of the fixed AXI mover; no synthesis or board execution."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, os, shutil, subprocess, time, signal
ROOT=Path(__file__).resolve().parents[1]
FILES=('gate_data_mover.sv','gate_mover_axi_memory.sv','gate_data_mover_tb.sv')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args()
    if not a.execute:
        print(json.dumps({'execute':False,'cases':13,'rtl':FILES,'hls_or_synthesis':False}));return
    if os.name=='nt':raise RuntimeError('Use configured WSL Python/tools')
    pause=ROOT/'results/goal2/pause.request'
    if pause.exists():raise RuntimeError('Pause marker present')
    cfg=json.loads((ROOT/'configs/project.json').read_text(encoding='utf-8-sig'))
    tools=Path(cfg['hardware']['hls']['vivado_executable']).parent
    tag=datetime.now(timezone.utc).isoformat().replace(':','_').replace('.','_').replace('+','_')
    out=ROOT/'results/goal4/data_mover'/tag;out.mkdir(parents=True)
    for f in FILES:shutil.copyfile(ROOT/'hardware'/f,out/f)
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    (out/'run.tcl').write_text('run all\nquit\n')
    record={'state':'running','passed':False,'recorded_utc':datetime.now(timezone.utc).isoformat(),
            'tools_directory':str(tools),'clock_ns':5,'source_files':list(FILES),
            'synthesis_performed':False,'cycle_boundary':'Accepted cmd_valid && cmd_ready edge to first rsp_valid edge; excludes response-consumer stalls after completion','memory_model':'AXI32,address64; one outstanding per direction; deterministic address/data/response stalls, no physical DDR/NoC model','commands':[]}
    def save(): (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    commands=[['xvlog','--sv',*FILES],['xelab','--debug','off','gate_data_mover_tb','-s','gate_data_mover_sim'],['xsim','gate_data_mover_sim','-tclbatch','run.tcl']]
    env=os.environ.copy();env['PATH']=str(tools)+os.pathsep+env.get('PATH','')
    try:
        for c in commands:
            if pause.exists():raise RuntimeError('Pause marker present')
            cmd=[str(tools/c[0]),*c[1:]];record['commands'].append(cmd);save()
            with (out/(c[0]+'.stdout.log')).open('w') as log:
                process=subprocess.Popen(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                begin=time.monotonic()
                while process.poll() is None:
                    if pause.exists() or time.monotonic()-begin>180:
                        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
                        raise RuntimeError('Pause/timeout at '+c[0])
                    time.sleep(.25)
            if process.returncode:raise RuntimeError(c[0]+' failed; see saved log')
        text=(out/'xsim.stdout.log').read_text(errors='replace')
        rows=[json.loads(s.split(' ',1)[1]) for s in text.splitlines() if s.startswith('GATE_MOVER_CASE ')]
        if len(rows)!=13 or len({r['id'] for r in rows})!=13 or 'PASS gate_data_mover 13 cases' not in text:raise RuntimeError('Missing focused validation evidence')
        for row in rows:
            row['read_bytes']=row['read_beats']*4;row['write_bytes']=row['write_beats']*4
        record.update(state='returned',passed=True,cases=rows,case_count=13,
                      boundary='Mover functional/cycle evidence under a bounded stall model; no synthesis, integrated detector timing, physical DDR, host cost, or board measurement')
        save();print(json.dumps({'passed':True,'evidence':str(out/'run.json'),'cases':rows},indent=2))
    except Exception as exc:
        record.update(state='stopped',error=str(exc));save();raise
if __name__=='__main__':main()
