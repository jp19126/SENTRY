"""Replay exact HLS RTL using a compact simulation-only AXI memory fixture."""
from pathlib import Path
import argparse,json,re,shutil,struct,subprocess,time,os,signal,sys,shlex
from datetime import datetime,timezone
from hardware import ROOT,digest,write_json

def qualify_runtime(hls,out,export,cases):
 """Native-qualify nominated runtime and two explicit decision-boundary calls."""
 hls=Path(hls).resolve();out=Path(out).resolve();export=Path(export).resolve();cases=Path(cases).resolve();out.mkdir(exist_ok=False)
 manifest=json.loads((export/'manifest.json').read_text())
 if digest(export/'model.bin')!=manifest['model_sha256']:raise ValueError('Export identity changed')
 shutil.copyfile(export/'model.bin',out/'runtime_model.bin');shutil.copyfile(cases,out/'base_cases.bin')
 exe=hls/'cnn_cosim_reference'
 args=[str(exe),str(out/'runtime_model.bin'),str(out/'base_cases.bin')]
 first=subprocess.run(args,capture_output=True,text=True,timeout=120)
 write_json(out/'base_native.json',{'command':args,'returncode':first.returncode,'stdout':first.stdout,'stderr':first.stderr});first.check_returncode()
 margin=int(next(line for line in first.stdout.splitlines() if line.startswith('0 ')).split()[1])
 original=cases.read_bytes();n=struct.unpack_from('<I',original)[0]
 nw=struct.unpack_from('<I',original,4)[0]
 if nw!=1:raise ValueError('Boundary fixture requires first case to be one complete window')
 ids_masks=original[36:36+768]
 expanded=struct.pack('<I',n+2)+original[4:]
 for threshold,blocked in [(margin,0),(margin-1,1)]:expanded+=struct.pack('<IIIiIiii',1,1,1,0,0,threshold,margin,blocked)+ids_masks
 (out/'runtime_cases.bin').write_bytes(expanded)
 args=[str(exe),str(out/'runtime_model.bin'),str(out/'runtime_cases.bin')];run=subprocess.run(args,capture_output=True,text=True,timeout=120)
 write_json(out/'cosim_runtime.json',{'export':str(export),'export_manifest_sha256':digest(export/'manifest.json'),'model_sha256':digest(out/'runtime_model.bin'),'cases_source':str(cases),'base_cases_sha256':digest(cases),'cases_sha256':digest(out/'runtime_cases.bin'),'kernel_sha256':digest(hls/'source/cnn.cpp'),'native_executable_sha256':digest(exe),'native_command':args,'native_returncode':run.returncode,'native_stdout':run.stdout,'native_stderr':run.stderr,'boundary_calls':[{'threshold':margin,'blocked':0},{'threshold':margin-1,'blocked':1}],'boundary':'FIT architectural verification; invented threshold boundary cases do not calibrate or evaluate detector quality'})
 run.check_returncode();print(run.stdout)


def prepare(hls,out,registers,runtime=None):
 hls=Path(hls).resolve();runtime=Path(runtime).resolve() if runtime else hls;out=Path(out).resolve();out.mkdir(exist_ok=False)
 rt=json.loads((runtime/'cosim_runtime.json').read_text())
 if rt['native_returncode']!=0 or digest(runtime/'runtime_model.bin')!=rt['model_sha256'] or digest(runtime/'runtime_cases.bin')!=rt['cases_sha256']:raise ValueError('Native-qualified runtime identity changed')
 native={int(line.split()[0]):int(line.split()[1]) for line in rt['native_stdout'].splitlines() if line and line[0].isdigit()}
 rtl=hls/'project/solution/syn/verilog';source=(rtl/'kd_cnn_top.v').read_text()
 params=re.findall(r'^parameter\s+(C_\w+)\s*=\s*([^;]+);',source,re.M)
 ports=re.findall(r'^(input|output)\s+(\[[^\]]+\])?\s*(\w+)\s*;',source,re.M)
 if not {'ap_clk','ap_rst_n','s_axi_control_AWADDR','m_axi_output_r_WDATA'}<=set(n for _,_,n in ports):raise ValueError('Unexpected generated port declaration')
 lines=['`timescale 1ns/1ps','module replay_tb;']+[f'localparam {k} = {v};' for k,v in params]
 for direction,width,name in ports:
  kind='reg' if direction=='input' and (name.startswith('s_axi_') or name.startswith('ap_')) else 'wire'
  lines.append(f'{kind} {width or ""} {name};')
 lines+=['kd_cnn_top dut('+','.join(f'.{n}({n})' for _,_,n in ports)+');']
 memports='ARVALID ARREADY ARADDR ARID ARLEN ARSIZE ARBURST RVALID RREADY RDATA RLAST RID RRESP AWVALID AWREADY AWADDR AWID AWLEN AWSIZE AWBURST WVALID WREADY WDATA WSTRB WLAST BVALID BREADY BID BRESP'.split()
 for name,width,size in [('model',32,2040000),('input_r',32,16384),('output_r',256,4096)]:
  lines.append(f'axi_memory #(.DW({width}),.SIZE({size})) {name}_mem(.clk(ap_clk),.resetn(ap_rst_n),'+','.join(f'.{p}(m_axi_{name}_{p})' for p in memports)+');')
  lines += [f'assign m_axi_{name}_{p}=0;' for p in ['RUSER','BUSER']]
 lines += [r'''
longint unsigned cycles=0,start_cycle;
always #2.5 ap_clk=~ap_clk;
always @(posedge ap_clk) begin cycles<=cycles+1;if(cycles>60000000)$fatal(1,"Cycle ceiling");end
task write_reg(input [6:0] addr,input [31:0] value);
 begin
  @(negedge ap_clk);s_axi_control_AWADDR=addr;s_axi_control_AWVALID=1;
  do @(posedge ap_clk); while(!s_axi_control_AWREADY);
  @(negedge ap_clk);s_axi_control_AWVALID=0;s_axi_control_WDATA=value;s_axi_control_WSTRB=15;s_axi_control_WVALID=1;
  do @(posedge ap_clk); while(!s_axi_control_WREADY);
  @(negedge ap_clk);s_axi_control_WVALID=0;
  do @(posedge ap_clk); while(!s_axi_control_BVALID);
  if(s_axi_control_BRESP!=0)$fatal(1,"AXI-lite write response");
 end
endtask
task read_reg(input [6:0] addr,output [31:0] value);
 begin
  @(negedge ap_clk);s_axi_control_ARADDR=addr;s_axi_control_ARVALID=1;
  do @(posedge ap_clk); while(!s_axi_control_ARREADY);
  @(negedge ap_clk);s_axi_control_ARVALID=0;
  do @(posedge ap_clk); while(!s_axi_control_RVALID);
  if(s_axi_control_RRESP!=0)$fatal(1,"AXI-lite read response");
  value=s_axi_control_RDATA;
 end
endtask
function [31:0] result_word(input integer index);
 result_word={output_r_mem.mem[index*4+3],output_r_mem.mem[index*4+2],output_r_mem.mem[index*4+1],output_r_mem.mem[index*4]};
endfunction
integer fd,nbytes;reg [31:0] status;
initial begin
''']
 for direction,width,name in ports:
  if direction=='input' and (name.startswith('s_axi_') or name.startswith('ap_')):lines.append(name+'=0;')
 lines += ['s_axi_control_BREADY=1;s_axi_control_RREADY=1;', 'repeat(5)@(negedge ap_clk);', 'fd=$fopen("model.bin","rb");if(!fd)$fatal(1,"Missing model");nbytes=$fread(model_mem.mem,fd);$fclose(fd);if(nbytes!=2036944)$fatal(1,"Model size");', 'ap_rst_n=1;repeat(5)@(negedge ap_clk);']
 reg=json.loads(Path(registers).read_text())['registers']
 fixed={'blob_1':0,'blob_2':0,'bytes':2036944,'ids_1':0,'ids_2':0,'masks_1':8192,'masks_2':0,'output_r_r_1':0,'output_r_r_2':0}
 lines += [f"write_reg({reg[k]},32'h{v&0xffffffff:08x});" for k,v in fixed.items()]
 cases=(runtime/'runtime_cases.bin').read_bytes();count=struct.unpack_from('<I',cases)[0];offset=4;expected=[]
 for i in range(count):
  nw,first,final,prev,pw,threshold,maximum,blocked=struct.unpack_from('<IIIiIiii',cases,offset);offset+=32
  data=bytearray(16384);data[:nw*512]=cases[offset:offset+nw*512];offset+=nw*512;data[8192:8192+nw*256]=cases[offset:offset+nw*256];offset+=nw*256
  (out/f'input{i}.hex').write_text(''.join(f'{b:02x}\n' for b in data))
  lines += [f'$readmemh("input{i}.hex",input_r_mem.mem);']
  for key,val in {'count':nw,'load':int(i==0),'first':first,'final_r':final,'previous':prev,'previous_windows':pw,'threshold':threshold}.items():lines.append(f"write_reg({reg[key]},32'h{val&0xffffffff:08x});")
  lines += ['start_cycle=cycles;write_reg(0,1);status=0;', 'while(!status[1])begin repeat(256)@(posedge ap_clk);read_reg(0,status);end', f'$display("CASE {i} cycles=%0d margin=%0d maximum=%0d blocked=%0d windows=%0d error=%0d valid=%0d",cycles-start_cycle,$signed(result_word(0)),$signed(result_word(1)),result_word(2),result_word(4),result_word(3),result_word(5));']
  lines += [f"if(result_word(0)!==32'h{native[i]&0xffffffff:08x} || result_word(1)!==32'h{maximum&0xffffffff:08x} || result_word(2)!=={blocked} || result_word(4)!=={nw if first else pw+nw} || result_word(3)!==0 || result_word(5)!=={final})$fatal(1,\"CASE {i} mismatch\");"]
  expected.append({'case':i,'margin':native[i],'maximum':maximum,'blocked':blocked,'windows':nw if first else pw+nw,'valid':final})
 if offset!=len(cases):raise ValueError('Trailing cases bytes')
 lines += ['$display("RTL_PASS cases=%0d total_cycles=%0d model_read_beats=%0d input_read_beats=%0d output_write_beats=%0d",'+str(count)+',cycles,model_mem.read_beats,input_r_mem.read_beats,output_r_mem.write_beats);$finish;end','endmodule']
 (out/'replay_tb.sv').write_text('\n'.join(lines)+'\n')
 shutil.copyfile(ROOT/'hardware/kd_cnn_v2/axi_memory.sv',out/'axi_memory.sv');shutil.copyfile(runtime/'runtime_model.bin',out/'model.bin')
 rtlout=out/'rtl';shutil.copytree(rtl,rtlout)
 for p in rtl.glob('*.dat'):shutil.copyfile(p,out/p.name)
 vivado=Path('/home/jp19126/Xilinx/2025.2/Vivado');commands=[['xvlog','--sv',*[str(p) for p in sorted(rtlout.glob('*.v'))],str(out/'axi_memory.sv'),str(out/'replay_tb.sv'),str(vivado/'data/verilog/src/glbl.v')],['xelab','-L','unisims_ver','-L','unimacro_ver','-L','xpm','replay_tb','glbl','-s','replay'],['xsim','replay','-runall']]
 script='#!/bin/bash\nset -e\nsource '+shlex.quote(str(vivado/'settings64.sh'))+'\n'+'\n'.join(shlex.join(c) for c in commands)+'\n';(out/'run.sh').write_text(script)
 write_json(out/'prepared.json',{'hls':str(hls),'runtime':str(runtime),'runtime_record_sha256':digest(runtime/'cosim_runtime.json'),'model_sha256':digest(out/'model.bin'),'cases_sha256':digest(runtime/'runtime_cases.bin'),'run_script_sha256':digest(out/'run.sh'),'testbench_sha256':digest(out/'replay_tb.sv'),'memory_model_sha256':digest(out/'axi_memory.sv'),'rtl_sha256':{p.name:digest(p) for p in rtlout.glob('*') if p.is_file()},'expected':expected,'commands':commands,'boundary':'Exact generated RTL; simple one-outstanding AXI backpressure fixture, not physical DDR or device latency'})

def execute(out,seconds):
 out=Path(out).resolve();record=out/'run.json'
 if record.exists():raise FileExistsError('Preserve prior attempt')
 if not 1<=seconds<=7200:raise ValueError('Bound run')
 r=json.loads((out/'prepared.json').read_text())
 if 'run_script_sha256' in r and digest(out/'run.sh')!=r['run_script_sha256']:raise ValueError('Replay commands changed')
 for name,key in [('model.bin','model_sha256'),('replay_tb.sv','testbench_sha256'),('axi_memory.sv','memory_model_sha256')]:
  if digest(out/name)!=r[key]:raise ValueError('Fixture changed')
 for name,h in r['rtl_sha256'].items():
  if digest(out/'rtl'/name)!=h:raise ValueError('RTL changed')
 sys.path.insert(0,str(ROOT/'scripts'));from r1_challengers import resource_check
 log={'state':'running','start_utc':datetime.now(timezone.utc).isoformat(),'preflight':resource_check(True)};start=time.monotonic();proc=None
 try:
  with (out/'stdout.log').open('x') as f:
   proc=subprocess.Popen(['bash',str(out/'run.sh')],cwd=out,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);log['pid']=proc.pid;write_json(record,log)
   while proc.poll() is None:
    if (out/'STOP_REQUESTED').exists():raise InterruptedError('Coordinator stop')
    if time.monotonic()-start>seconds:raise TimeoutError('RTL replay ceiling')
    obs=resource_check()
    with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(obs)+'\n')
    time.sleep(5)
   passed=proc.returncode==0 and 'RTL_PASS' in (out/'stdout.log').read_text()
   log.update(returncode=proc.returncode,state='completed' if passed else 'failed')
 except BaseException as e:log.update(state='failed',error=repr(e));raise
 finally:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
  log['seconds']=time.monotonic()-start;write_json(record,log)
 if log['state']!='completed':raise RuntimeError('RTL replay failed; inspect retained log')

def main():
 p=argparse.ArgumentParser();p.add_argument('--hls',type=Path);p.add_argument('--runtime',type=Path);p.add_argument('--qualify-runtime',action='store_true');p.add_argument('--export',type=Path);p.add_argument('--cases',type=Path);p.add_argument('--registers',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--seconds',type=int,default=3600);a=p.parse_args()
 if a.qualify_runtime:qualify_runtime(a.hls,a.out,a.export,a.cases);return
 if not (a.out/'prepared.json').exists():prepare(a.hls,a.out,a.registers,a.runtime)
 if a.execute:execute(a.out,a.seconds)
if __name__=='__main__':main()
