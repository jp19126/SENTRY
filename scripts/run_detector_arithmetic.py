"""Connect the fixed sequencer/mover to unchanged complete HLS arithmetic tops.

The shared byte arena is simulation memory, never a board allocation. Default
prepares source only; --execute compiles/runs bounded offline RTL. --stop-after
permits first-connected-service inspection before paying for a full detector.
"""
from pathlib import Path
import argparse, json, re, shutil, subprocess, time, os, signal, copy
from datetime import datetime, timezone
from plan_detector_commands import create_plan, write_rtl_plan
ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT/'results/goal4/internal_integration/arena_w8_qat_real_input_v1'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def need(ok, message):
    if not ok: raise ValueError(message)
def save(p, data): p.write_text(json.dumps(data, indent=2)+'\n')

def diagnostic_prefix_plan(original, rows):
    """Explicit partial integration, full256-wide arithmetic on first saved rows."""
    need(rows==16, 'Only the purpose-approved16-row integration diagnostic is staged')
    plan=copy.deepcopy(original)
    keep={'validate_window_and_load_inputs','gather_word_embeddings','gather_type_embeddings',
          'embedding_sum_rows0','embedding_ln_rows0','layer0_query_quantize_rows0',
          'layer0_query','layer0_query_rescale_rows0'}
    # The third movement expands the mask; position embeddings are read directly.
    selected=[copy.deepcopy(original['commands'][0])]+[copy.deepcopy(c) for c in original['commands'][1:4]]
    selected += [copy.deepcopy(c) for c in original['commands'] if c['kind']=='service' and c['name'] in keep]
    need(len(selected)==9 and sum(c['kind']=='service' for c in selected)==5,'Actual prefix descriptor association changed')
    for index,c in enumerate(selected):
        c['original_command_index']=c['index'];c['index']=index
        if c['kind']=='movement':
            c['rows']=rows;c['useful_read_bytes']=c['useful_write_bytes']=rows*c['columns']*4
            c['destination']['extent_bytes']=rows*c['columns']*4
        elif c['kind']=='service':
            c['arguments']['ROWS']=rows
            for write in c.get('register_writes',[]):
                if write.get('argument')=='ROWS':write['value']=rows
    plan.update(identity=original['identity']+'_connected_prefix16_v1',commands=selected,
        status='Explicit reduced-row diagnostic; not a complete detector or original full command schedule',
        diagnostic=dict(rows=rows,original_plan_identity=original['identity'],
            scope='Actual gather -> embedding sum -> LN -> A8 -> integer query -> rescale, first16 saved rows, original width/weights/scales',
            integer_calls=1,fp32_calls=4,stop_after=5,complete_detector=False),
        counts={'commands':9,'integer_calls':1,'fp32_calls':4})
    return plan


def generate(plan, out, prep):
    """Preserve actual generated port maps from already validated component TBs."""
    write_rtl_plan(plan, out)
    linear = (ROOT/'hardware/linear_axi_tb.sv').read_text()
    fp = (ROOT/'hardware/fixed_fp32_axi_tb.sv').read_text()
    control = (ROOT/'hardware/gate_detector_control_tb.sv').read_text()
    shared = linear[:linear.index('module linear_axi_tb;')]
    shared = shared.replace('byte unsigned mem [0:DEPTH-1];', '')
    shared = shared.replace('bit written [0:DEPTH-1];', '')
    shared = re.sub(r'\bmem\[', 'gate_detector_arithmetic_tb.arena[', shared)
    shared = re.sub(r'\s*written\[word_base\+lane\] = 1;', '', shared)
    (out/'shared_arithmetic_memory.sv').write_text(shared)
    mover_memory = (ROOT/'hardware/gate_mover_axi_memory.sv').read_text()
    mover_memory = mover_memory.replace('byte unsigned mem[0:DEPTH-1];', '')
    mover_memory = re.sub(r'\bmem\[', 'gate_detector_arithmetic_tb.arena[', mover_memory)
    (out/'shared_mover_memory.sv').write_text(mover_memory)
    # The real sequencer and mover have exactly the same public connections as
    # their existing control-only integration. The arithmetic stubs are excluded.
    declarations = control[control.index(' logic  clk;'):control.index(' logic [2:0] fault_mode;')]
    interconnect = control[control.index(' gate_detector_sequencer dut(.*);'):control.index(' gate_linear_top_control_s_axi registers_0(')]
    interconnect = interconnect.replace('gate_mover_axi_memory #(.DEPTH(DEPTH),.BASE(BASE),.STALL(0),.PIPELINED_READ(1)) memory(.*);',
        'gate_mover_axi_memory #(.DEPTH(DEPTH),.BASE(BASE),.STALL(0),.PIPELINED_READ(1)) memory(.*);')
    body = '''`timescale 1ns/1ps
module gate_detector_arithmetic_tb;
 localparam integer TOKENS=256;
 localparam integer DEPTH=67108864;
 localparam [63:0] BASE=0;
 byte unsigned arena[0:DEPTH-1];
''' + declarations + '''
 logic [2:0] fault_mode=0;
 logic [1:0] slave_awready,slave_wready,slave_bvalid,slave_arready,slave_rvalid;
 logic [1:0] slave_bresp[0:1],slave_rresp[0:1];
 logic [31:0] slave_rdata[0:1];
 assign ctl_awready=slave_awready[ctl_target];
 assign ctl_wready=slave_wready[ctl_target];
 assign ctl_bvalid=slave_bvalid[ctl_target];
 assign ctl_bresp=slave_bresp[ctl_target];
 assign ctl_arready=slave_arready[ctl_target];
 assign ctl_rvalid=slave_rvalid[ctl_target];
 assign ctl_rresp=slave_rresp[ctl_target];
 assign ctl_rdata=slave_rdata[ctl_target];
''' + interconnect
    for index, text, module, ports in [(0,linear,'gate_linear_top',['a','w','o']),
                                      (1,fp,'gate_fixed_fp32_top',['x','y','z','accumulators','output','codes'])]:
        instance = re.search(r'    '+module+r' dut \((.*?)\n    \);', text, re.S)
        need(instance is not None, 'Validated component port map missing: '+module)
        connection = instance.group(1)
        for port in ports:
            connection = re.sub(r'\b'+port+r'_bus\b', f'e{index}_{port}_bus', connection)
            body += f' gate_axi_bus e{index}_{port}_bus();\n'
            body += f' gate_bounded_axi_memory #(.DEPTH(DEPTH),.READ_ALLOWED(1),.WRITE_ALLOWED(1),.NAME("e{index}_{port}")) e{index}_{port}_memory(clk,resetn,DEPTH,e{index}_{port}_bus);\n'
        for signal_name in ['awready','wready','bvalid','bresp','arready','rvalid','rdata','rresp']:
            connection = connection.replace('c_'+signal_name, f'slave_{signal_name}[{index}]')
        for signal_name in ['awvalid','wvalid','bready','arvalid','rready']:
            connection = connection.replace('c_'+signal_name, f'ctl_{signal_name} && ctl_target=={index}')
        for signal_name in ['awaddr','araddr','wdata','wstrb']:
            replacement='ctl_'+signal_name
            if index == 0 and signal_name in ['awaddr','araddr']: replacement += '[6:0]'
            connection = connection.replace('c_'+signal_name, replacement)
        connection = connection.replace('.interrupt(interrupt)', f'.interrupt(engine_interrupt[{index}])')
        connection = connection.replace('.interrupt()', f'.interrupt(engine_interrupt[{index}])')
        body += ' '+module+f' arithmetic_{index} ('+connection+'\n );\n'
    if plan.get('diagnostic'):
        body += """
 logic [7:0] s_axi_awaddr,s_axi_araddr;
 logic [31:0] s_axi_wdata,s_axi_rdata;
 logic [3:0] s_axi_wstrb;
 logic s_axi_awvalid,s_axi_awready,s_axi_wvalid,s_axi_wready,s_axi_bvalid,s_axi_bready;
 logic s_axi_arvalid,s_axi_arready,s_axi_rvalid,s_axi_rready;
 logic [1:0] s_axi_bresp,s_axi_rresp;
 gate_host_mailbox #(.TOKENS(TOKENS)) host_mailbox(.*);
 task host_write(input logic [7:0] address,input logic [31:0] value);
  bit aw_done,w_done;
  begin
   @(negedge clk);s_axi_awaddr=address;s_axi_wdata=value;s_axi_awvalid=1;s_axi_wvalid=1;
   aw_done=0;w_done=0;
   while(!aw_done || !w_done)begin
    @(posedge clk);
    if(s_axi_awvalid&&s_axi_awready)aw_done=1;
    if(s_axi_wvalid&&s_axi_wready)w_done=1;
    @(negedge clk);s_axi_awvalid=!aw_done;s_axi_wvalid=!w_done;
   end
   s_axi_bready=1;
   do @(posedge clk);while(!s_axi_bvalid);
   if(s_axi_bresp!=0)$fatal(1,"Mailbox AXI-Lite write refused %h",address);
   $display("GATE_CONNECTED_HOST_WRITE address=%h value=%h",address,value);
   @(negedge clk);s_axi_bready=0;
  end
 endtask
"""
    captures=[]
    for command_index, command in enumerate(plan['commands']):
        if command['kind']!='service':continue
        a=command['arguments']; integer=command['engine']=='integer'
        quant=not integer and a['OP']==1
        offset=a['CODES' if quant else 'OUTPUT_R']['arena_offset']
        count=a['ROWS']*(a['OUTPUTS'] if integer or a['OP']==0 else a['WIDTH'])
        index=len(captures)+1
        if integer:
            input_sizes={'ACTIVATIONS':a['ROWS']*a['INNER'],'WEIGHTS':a['INNER']*a['OUTPUTS']}
        else:
            r,w,n,op=a['ROWS'],a['WIDTH'],a['OUTPUTS'],a['OP']
            input_sizes={'X':r*w*4}
            if op==0:input_sizes.update(Y=n*w*4,**({'Z':n*4} if a['DOT_BIAS'] else {}))
            elif op==2:input_sizes={'ACCUMULATORS':r*w*4,'Y':w*4,'Z':w*4}
            elif op==3:input_sizes.update(Y=r*w*4,Z=r*w*4)
            elif op in (4,6):input_sizes.update(Y=r*w*4)
            elif op==5:input_sizes.update(Y=w*4,Z=w*4)
        inputs=[]
        for argument,extent in input_sizes.items():
            location=a[argument]
            if 'select_by_frozen_precision_group' in location:location=location['alternatives']['8']
            inputs.append(dict(argument=argument,offset=location['arena_offset'],bytes=extent,file=f'service_{index:04d}_{argument.lower()}.bin'))
        captures.append(dict(service_index=index,command_index=command_index,name=command['name'],engine=command['engine'],op=a.get('OP'),arguments=a,inputs=inputs,offset=offset,bytes=count*(1 if quant else 4),file=f'service_{index:04d}.bin'))
    save(out/'service_manifest.json',dict(plan_identity=plan['identity'],captures=captures))
    body += ' task dump_range(input string name,input integer begin_byte,input integer byte_count);integer fd;begin\n fd=$fopen(name,\"wb\");if(fd==0)$fatal(1,\"Cannot capture input\");\n for(integer k=0;k<byte_count;k=k+1)$fwrite(fd,\"%c\",arena[begin_byte+k]);$fclose(fd);end endtask\n'
    body += ' task capture_inputs(input integer index);begin case(index)\n'
    for capture in captures:
        body += f" {capture['service_index']}:begin\n"
        for item in capture['inputs']:
            body += f" dump_range(\"{item['file']}\",{item['offset']},{item['bytes']});\n"
        body += ' end\n'
    body += ' default:$fatal(1,\"Invalid input service index\");endcase end endtask\n'
    body += ' task capture_service(input integer index);\n integer file_handle, begin_byte, byte_count;string name;begin\n case(index)\n'
    for capture in captures:
        body += f" {capture['service_index']}:begin begin_byte={capture['offset']};byte_count={capture['bytes']};name=\"{capture['file']}\";end\n"
    body += ' default:$fatal(1,\"Invalid completed service\");\n endcase\n file_handle=$fopen(name,\"wb\");if(file_handle==0)$fatal(1,\"Cannot capture actual output\");\n for(integer k=0;k<byte_count;k=k+1)$fwrite(file_handle,\"%c\",arena[begin_byte+k]);\n $fclose(file_handle);end endtask\n'
    body += r'''
 longint unsigned cycle=0, accepted_cycle=0;
 integer stop_after=0, completed_services=0, fd, n;
 longint unsigned timeout_cycles=300000000;
 string image_path;
 logic [15:0] previous_command;
 always #2.5 clk=~clk;
 always @(posedge clk) if(resetn) begin
  cycle<=cycle+1;
  if(window_valid&&window_ready)accepted_cycle=cycle;
  if(cycle>timeout_cycles)$fatal(1,"Integrated real-arithmetic timeout command=%0d services=%0d",command_index,completed_services);
  if(dut.state==9 && (dut.descriptor[31:0]==1 || dut.descriptor[31:0]==2))capture_inputs(completed_services+1);
  if(dut.state==18 && dut.bus_rsp_valid && dut.bus_rsp_ready) begin
   completed_services=completed_services+1;
   capture_service(completed_services);
   $display("GATE_CONNECTED_SERVICE {\"completed\":%0d,\"command\":%0d,\"cycle\":%0d}",completed_services,command_index,cycle);
   if(stop_after>0 && completed_services==stop_after)begin
    fd=$fopen("arena_after.bin","wb");
    for(integer i=0;i<DEPTH;i=i+1)$fwrite(fd,"%c",arena[i]);
    $fclose(fd);
    $display("GATE_CONNECTED_PARTIAL {\"completed_services\":%0d,\"elapsed_cycles\":%0d}",completed_services,cycle-accepted_cycle);
    $finish;
   end
  end
 end
 initial begin
  if(!$value$plusargs("IMAGE=%s",image_path))$fatal(1,"IMAGE required");
  n=$value$plusargs("STOP_AFTER=%d",stop_after);
  n=$value$plusargs("TIMEOUT_CYCLES=%d",timeout_cycles);
  fd=$fopen(image_path,"rb");if(fd==0)$fatal(1,"Cannot open prepared arena");
  n=$fread(arena,fd);$fclose(fd);if(n!=DEPTH)$fatal(1,"Arena extent differs");
  clk=0;resetn=0;config_valid=0;window_valid=0;result_ready=0;
  repeat(5)@(negedge clk);resetn=1;
  config_w8_mask=16'hffff;config_threshold_bits=32'hTHRESHOLD;config_arena_base=BASE;config_model_id=64'hbf04010000000001;config_valid=1;
  @(posedge clk);if(!config_ready)$fatal(1,"Configuration refused");@(negedge clk);config_valid=0;
  wait(window_ready);@(negedge clk);
  window_document_id=77;window_sample_id=1;window_index=0;window_last=1;window_arena_base=BASE;window_valid_length=VALID_LENGTH;window_valid=1;
  @(posedge clk);if(!window_ready)$fatal(1,"Window refused");@(negedge clk);window_valid=0;
  wait(result_valid);@(negedge clk);
  if(result_status!=0 || integer_calls!=24 || fp32_calls!=411 || completed_services!=435)$fatal(1,"Complete detector failed status=%0d i=%0d f=%0d",result_status,integer_calls,fp32_calls);
  if(result_sample_id!=1 || result_document_id!=77 || result_window_index!=0 || !result_last || result_max_bits!=result_risk_bits)$fatal(1,"Result association/aggregation failed");
  if(result_reject!=(result_risk_bits>config_threshold_bits))$fatal(1,"Strict threshold failed");
  fd=$fopen("arena_after.bin","wb");for(integer i=0;i<DEPTH;i=i+1)$fwrite(fd,"%c",arena[i]);$fclose(fd);
  $display("GATE_CONNECTED_RESULT {\"status\":%0d,\"risk_bits\":\"%h\",\"reject\":%0d,\"elapsed_cycles\":%0d,\"integer_calls\":%0d,\"fp32_calls\":%0d}",result_status,result_risk_bits,result_reject,cycle-accepted_cycle,integer_calls,fp32_calls);
  $finish;
 end
endmodule
'''.replace('THRESHOLD',prep['threshold_bits_hex']).replace('VALID_LENGTH',str(prep['valid_length']))
    if plan.get('diagnostic'):
        body=body.replace('clk=0;resetn=0;config_valid=0;window_valid=0;result_ready=0;',
            "clk=0;resetn=0;s_axi_awvalid=0;s_axi_wvalid=0;s_axi_bready=0;s_axi_arvalid=0;s_axi_rready=0;s_axi_awaddr=0;s_axi_araddr=0;s_axi_wdata=0;s_axi_wstrb=4'hf;")
        first=body.index("  config_w8_mask=16'hffff;")
        last=body.index('  wait(result_valid);',first)
        body=body[:first]+f"""
  host_write('h10,32'hffff);host_write('h14,32'h{prep['threshold_bits_hex']});
  host_write('h18,0);host_write('h1c,0);host_write('h20,1);host_write('h24,32'hbf040100);
  host_write('h00,1);wait(window_ready);
  host_write('h30,1);host_write('h34,0);host_write('h38,77);host_write('h3c,0);
  host_write('h40,0);host_write('h44,{prep['valid_length']});host_write('h48,1);host_write('h00,2);
"""+body[last:]
        shutil.copyfile(ROOT/'hardware/gate_host_mailbox.sv',out/'gate_host_mailbox.sv')
    (out/'gate_detector_arithmetic_tb.sv').write_text(body)
    for name in ['gate_detector_sequencer.sv','gate_data_mover.sv']:
        shutil.copyfile(ROOT/'hardware'/name, out/name)

def integration_resources():
    """Observe actual host headroom for this bounded offline integration only."""
    counters={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,tail=line.split(':',1)
        if key in ('MemFree','MemAvailable','Cached','SReclaimable','SwapFree'):
            counters[key]=int(tail.split()[0])
    host=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],capture_output=True,text=True,timeout=10)
    need(host.returncode==0,'Cannot observe Windows integration memory headroom')
    counters['windows_free_kib']=int(host.stdout.strip())
    counters['observed_utc']=datetime.now(timezone.utc).isoformat()
    return counters


def run_owned_command(command, out, env, record):
    """Bound this integration tool's process group and retain failed attempts."""
    observation=dict(tool=Path(command[0]).name,started_utc=datetime.now(timezone.utc).isoformat(),limit_seconds=900)
    record.setdefault('command_observations',[]).append(observation)
    record.update(state='running',current_command=command);save(out/'run.json',record)
    proc=None;started=time.monotonic();last_observe=0.
    try:
        need(set(range(6))<=os.sched_getaffinity(0),'Integration requires the assigned CPU0..5 partition')
        os.sched_setaffinity(0,set(range(6)))
        observation['preflight']=integration_resources()
        need(observation['preflight']['MemAvailable']>=8*1024*1024 and observation['preflight']['windows_free_kib']>=4*1024*1024,'Integration preflight needs8GiB WSL and4GiB Windows headroom')
        observation.update(runtime_wsl_floor_kib=4*1024*1024,runtime_windows_floor_kib=2*1024*1024)
        save(out/'run.json',record)
        with (out/(Path(command[0]).name+'.stdout.log')).open('w') as log:
            proc=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            observation['pid']=proc.pid;save(out/'run.json',record)
            while proc.poll() is None:
                if (ROOT/'results/goal2/pause.request').exists():raise RuntimeError('Pause marker present')
                if time.monotonic()-started>900:raise RuntimeError('Bounded900-second integration command limit')
                if time.monotonic()-last_observe>=5:
                    current=integration_resources()
                    with (out/'resource_observations.jsonl').open('a',encoding='utf-8') as history:
                        history.write(json.dumps(dict(current,tool=observation['tool'],assigned_process_group=proc.pid))+'\n')
                    observation['latest_memory_kib']=current;save(out/'run.json',record);last_observe=time.monotonic()
                    need(current['MemAvailable']>=4*1024*1024 and current['windows_free_kib']>=2*1024*1024,'Integration resource guard:4GiB WSL and2GiB Windows headroom required')
                if 'actual_affinity' not in observation:
                    observation['actual_affinity']=sorted(os.sched_getaffinity(proc.pid));save(out/'run.json',record)
                time.sleep(.5)
        if proc.returncode:raise RuntimeError('Tool failed; inspect '+str(out/(Path(command[0]).name+'.stdout.log')))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        record.update(state='stopped' if proc is not None and proc.returncode is not None and proc.returncode<0 else 'failed',error=str(exc))
        raise
    finally:
        observation.update(elapsed_seconds=time.monotonic()-started,returncode=proc.returncode if proc else None)
        save(out/'run.json',record)
    return observation['elapsed_seconds']


def run_existing_snapshot(config, build, stop_after):
    """One bounded capture in a new directory; never overwrite compiled evidence."""
    old=read(build/'run.json')
    need(old.get('state')=='compiled' and old.get('passed'),'Successful integrated compile required')
    source_matches={name:(build/name).read_bytes()==(ROOT/'hardware'/name).read_bytes()
                    for name in ['gate_detector_sequencer.sv','gate_data_mover.sv']+(['gate_host_mailbox.sv'] if (build/'gate_host_mailbox.sv').exists() else [])}
    # The coordinator explicitly authorized replay of this frozen validated L256
    # snapshot while new shape-source work proceeds. Report the distinction.
    need(not (ROOT/'results/goal2/pause.request').exists(),'Pause marker present')
    need(os.name!='nt','Run using WSL Linux Python')
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=ROOT/'results/goal4/real_arithmetic_integration'/tag;out.mkdir(parents=True)
    (out/'xsim.dir').symlink_to(build/'xsim.dir',target_is_directory=True)
    for name in ['xsim.ini','command_plan.mem','service_manifest.json']+(['diagnostic_plan.json'] if (build/'diagnostic_plan.json').exists() else []):
        shutil.copyfile(build/name,out/name)
    for source in build.glob('*.dat'):shutil.copyfile(source,out/source.name)
    (out/'run.tcl').write_text('run all\nquit\n')
    tools=Path(config['hardware']['hls']['vivado_executable']).parent
    command=[str(tools/'xsim'),'gate_detector_arithmetic_sim','-tclbatch','run.tcl',
             '-testplusarg','IMAGE='+str(PREPARED/'arena.bin'),'-testplusarg','STOP_AFTER='+str(stop_after)]
    record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),state='running',passed=False,
        compiled_build=str(build),compiled_record=old,stop_after=stop_after,complete_detector=stop_after==0,diagnostic=old.get('diagnostic'),diagnostic_plan=str(out/'diagnostic_plan.json') if old.get('diagnostic') else None,
        current_control_source_matches_compiled=source_matches,
        source_policy='frozen compiled L256 design; does not validate changed source or other lengths',
        timeout_seconds=900,launcher_affinity=sorted(os.sched_getaffinity(0)),command=command,arithmetic='actual unchanged synthesized services connected to sequencer/mover',
        memory='one shared simulation arena; nominal independent AXI timing; no physical measurement')
    save(out/'run.json',record)
    env=os.environ.copy();env['PATH']=str(tools)+os.pathsep+env.get('PATH','')
    record['simulation_elapsed_seconds']=run_owned_command(command,out,env,record)
    text=(out/'xsim.stdout.log').read_text(errors='replace')
    marker='GATE_CONNECTED_RESULT ' if stop_after==0 else 'GATE_CONNECTED_PARTIAL '
    observations=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith(marker)]
    record.update(state='returned',returncode=record['command_observations'][-1]['returncode'],passed=len(observations)==1,
                  observations=observations)
    save(out/'run.json',record)
    print(json.dumps(dict(passed=record['passed'],evidence=str(out/'run.json'),observations=observations),indent=2))
    need(record['passed'],'Inspect captured integration failure '+str(out/'xsim.stdout.log'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--run-only',action='store_true',help='Run existing compiled snapshot in a fresh capture directory; requires --execute')
    ap.add_argument('--stop-after',type=int,default=0)
    ap.add_argument('--diagnostic-prefix-rows',type=int,choices=[16],help='Purpose-approved partial mixed-service integration on first16 saved rows, full original width/weights')
    ap.add_argument('--output',type=Path,default=ROOT/'build/detector_arithmetic_static64')
    args=ap.parse_args()
    need(0<=args.stop_after<=435,'stop-after must be 0 (full) or 1..435')
    config=read(ROOT/'configs/project.json');plan=create_plan(config);prep=read(PREPARED/'preparation.json')
    need(prep['complete'] and prep['config_w8_mask']==65535 and prep['plan_identity']==plan['identity'],'Prepared W8 case differs')
    need((PREPARED/'arena.bin').stat().st_size==67108864,'Prepared image absent/truncated')
    out=args.output if args.output.is_absolute() else ROOT/args.output
    if args.diagnostic_prefix_rows:
        need(args.stop_after in (0,5),'Reduced diagnostic stops only after its five actual services')
        args.stop_after=5;plan=diagnostic_prefix_plan(plan,args.diagnostic_prefix_rows)
    if args.run_only:
        need(args.execute and not args.compile_only,'--run-only requires --execute and excludes --compile-only')
        run_existing_snapshot(config,out,args.stop_after);return
    need(not (out/'run.json').exists() or read(out/'run.json').get('state')=='prepared','Preserve prior compiled/executed sources; select a fresh --output or use --run-only')
    out.mkdir(parents=True,exist_ok=True);generate(plan,out,prep)
    if args.diagnostic_prefix_rows:save(out/'diagnostic_plan.json',plan)
    roots=[ROOT/'build/linear_hls_cached_static_bank/lanes64/project/solution/syn/verilog',
           ROOT/'build/fixed_fp32_hls/project/solution/syn/verilog']
    # Follow the recorded FP32 build, rather than a guessed folder.
    fpcontrol=ROOT/plan['fp32_build']['control_source']; roots[1]=fpcontrol.parent
    sources=[]
    for folder in roots:
        need(folder.is_dir(),'Missing synthesized RTL '+str(folder));sources += sorted(folder.glob('*.v'))
    sources=[out/name for name in ['shared_arithmetic_memory.sv','shared_mover_memory.sv','gate_detector_sequencer.sv','gate_data_mover.sv','gate_detector_arithmetic_tb.sv']+(['gate_host_mailbox.sv'] if args.diagnostic_prefix_rows else [])]+sources
    tools=Path(config['hardware']['hls']['vivado_executable']).parent
    fpbuild=read(ROOT/'results/goal4/fixed_fp32/run.json')
    tag=re.sub(r'[^A-Za-z0-9_-]','_',fpbuild['recorded_utc'])
    cached=ROOT/'build/fixed_fp32_rtl'/tag
    compiled=read(cached/'compiled.json')
    need(compiled['signature']==dict(synthesis_recorded_utc=fpbuild['recorded_utc'],service_schedule_revision=fpbuild['service_schedule_revision']),'FP32 compiled signature differs')
    for name in compiled['reachable_rtl_files']:
        need((cached/'rtl'/name).read_bytes()==(roots[1]/name).read_bytes(),'FP32 compiled RTL association differs: '+name)
    sim=cached/'ip_project/fixed_fp32_sim.sim/sim_1/behav/xsim'
    # Reuse only the already compiled official vendor library units, privately.
    # The new connected top is always freshly compiled and elaborated here.
    if args.execute and not (out/'vendor_libraries_copied.json').exists():
        shutil.copytree(sim/'xsim.dir',out/'xsim.dir',dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(compiled['snapshot'],'*.log','*.wdb'))
        shutil.copyfile(sim/'xsim.ini',out/'xsim.ini')
        save(out/'vendor_libraries_copied.json',dict(source=str(sim),signature=compiled['signature']))
    if args.execute:
        for path in roots[1].glob('*.dat'):shutil.copyfile(path,out/path.name)
    libraries=re.findall(r'-L ([A-Za-z0-9_]+)',(sim/'elaborate.sh').read_text())
    libargs=[word for name in dict.fromkeys(libraries) for word in ['-L',name]]
    commands=[[str(tools/'xvlog'),'--sv','--work','xil_defaultlib',*[str(p) for p in sources]],
              [str(tools/'xelab'),'--debug','off','--relax',*libargs,'xil_defaultlib.gate_detector_arithmetic_tb','xil_defaultlib.glbl','-s','gate_detector_arithmetic_sim']]
    (out/'run.tcl').write_text('run all\nquit\n')
    if not args.compile_only: commands += [[str(tools/'xsim'),'gate_detector_arithmetic_sim','-tclbatch','run.tcl','-testplusarg','IMAGE='+str(PREPARED/'arena.bin'),'-testplusarg','STOP_AFTER='+str(args.stop_after)]]
    record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),state='prepared',passed=False,
        arithmetic='actual unchanged synthesized integer static64 and FP32 tops',memory='shared 64MiB simulation arena; independent nominal AXI ports; no physical DDR/NoC model',
        complete_detector=args.stop_after==0,stop_after=args.stop_after,source_roots=[str(p) for p in roots],commands=commands,
        board_access=False,host_control='actual AXI-Lite mailbox configure/submit' if args.diagnostic_prefix_rows else 'direct sequencer handshake',source_plan=plan['identity'],prepared_arena=str(PREPARED),diagnostic=plan.get('diagnostic'),diagnostic_plan=str(out/'diagnostic_plan.json') if args.diagnostic_prefix_rows else None)
    save(out/'run.json',record)
    if not args.execute:print(json.dumps(record,indent=2));return
    need(os.name!='nt','Execute with WSL Linux Python')
    need(not (ROOT/'results/goal2/pause.request').exists(),'Pause marker present')
    env=os.environ.copy();env['PATH']=str(tools)+os.pathsep+env.get('PATH','')
    record['launcher_affinity']=sorted(os.sched_getaffinity(0));record['command_observations']=[]
    for command in commands:
        run_owned_command(command,out,env,record)
    record['compile_elapsed_seconds']=sum(x['elapsed_seconds'] for x in record['command_observations'] if x['tool']!='xsim')
    if not args.compile_only:record['simulation_elapsed_seconds']=next(x['elapsed_seconds'] for x in record['command_observations'] if x['tool']=='xsim')
    if not args.compile_only:
        text=(out/'xsim.stdout.log').read_text(errors='replace');marker='GATE_CONNECTED_RESULT ' if args.stop_after==0 else 'GATE_CONNECTED_PARTIAL '
        rows=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith(marker)]
        if len(rows)!=1:
            record.update(state='failed',passed=False,error='Missing successful real-arithmetic completion marker');save(out/'run.json',record);raise ValueError(record['error'])
        record['observation']=rows[0]
    record.update(state='compiled' if args.compile_only else 'returned',passed=True);save(out/'run.json',record)
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
