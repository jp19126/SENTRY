"""Prepare the complete CNN/CIPS/DDR connected VEK280 implementation.

This consumes an actually exported HLS IP. Preparation emits an exact Tcl and
register map; only --execute invokes Vivado, never a hardware server or board.
The existing installed board preset and established CIPS/NoC topology are reused.
"""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
from hardware import linux,digest,write_json,ROOT
def quote(x):
 s=linux(x)
 if any(c in s for c in '{}\r\n'):raise ValueError('Unsafe Tcl path')
 return '{'+s+'}'
def prepare(hls,out,config,implement):
 hls=Path(hls).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 component=hls/'project/solution/impl/ip/component.xml'
 tree=ET.parse(component);ns={'s':'http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009'}
 def text(tag):return tree.find('s:'+tag,ns).text
 vlnv=':'.join(text(k) for k in ['vendor','library','name','version'])
 interfaces=[n.find('s:name',ns).text for n in tree.findall('s:busInterfaces/s:busInterface',ns)]
 required=[]
 for candidates in [('m_axi_model',),('m_axi_input','m_axi_input_r'),('m_axi_output','m_axi_output_r'),('s_axi_control',)]:
  found=[n for n in candidates if n in interfaces]
  if len(found)!=1:raise ValueError(('Actual HLS interfaces differ',interfaces))
  required.append(found[0])
 registers={}
 for mm in tree.findall('s:memoryMaps/s:memoryMap',ns):
  for reg in mm.findall('s:addressBlock/s:register',ns):
   registers[reg.find('s:name',ns).text]=int(reg.find('s:addressOffset',ns).text,0)
 write_json(out/'hls_registers.json',{'vlnv':vlnv,'registers':registers,'component_sha256':digest(component)})
 cfg=json.loads(Path(config).read_text(encoding='utf-8-sig'));hw=cfg['hardware']
 preset_path=Path(hw['board_definition']['board_definition_path']).with_name('preset.xml')
 preset_tree=ET.parse(preset_path).getroot()
 xml=preset_tree.find(".//ip_preset[@preset_proc_name='ps_pmc_fixed_io_preset']//user_hier_parameter[@name='CONFIG.PS_PMC_CONFIG']")
 def xd(node):return {c.attrib['name']:xd(c) if list(c) else c.attrib['value'] for c in node}
 preset=xd(xml);preset.update(DESIGN_MODE='1',PS_USE_PMCPL_CLK0='1',PMC_CRP_PL0_REF_CTRL_FREQMHZ='200',PS_USE_M_AXI_FPD='1',PS_NUM_FABRIC_RESETS='1',PS_USE_FPD_CCI_NOC='1',PS_USE_NOC_LPD_AXI0='1',PMC_USE_PMC_NOC_AXI0='1')
 def nested(value):
  pairs=[]
  for k,v in value.items():
   p=str(k)+' '+('{'+nested(v)+'}' if isinstance(v,dict) else str(v));pairs.append(p if len(value)==1 else '{'+p+'}')
  return ' '.join(pairs)
 pd=' '.join('{'+k+'} {'+(nested(v) if isinstance(v,dict) else str(v))+'}' for k,v in preset.items())
 hook=out/'threads.tcl';hook.write_text('set_param general.maxThreads 2\nset_param synth.maxThreads 1\n')
 lines=['set_param general.maxThreads 2','set_param synth.maxThreads 1',
  f'create_project kd_cnn_system {quote(out/"project")} -part {hw["part"]}',
  f'set_property board_part {hw["board_definition"]["board_part"]} [current_project]',
  f'set_property ip_repo_paths {quote(component.parent)} [current_project]','update_ip_catalog','create_bd_design cnn_system',
  'set cips [create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips:3.4 cips]',f'set_property CONFIG.PS_PMC_CONFIG {{{pd}}} $cips',
  'set noc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 noc]',
  'set_property -dict [list CONFIG.NUM_SI 7 CONFIG.NUM_MI 0 CONFIG.NUM_MC 1 CONFIG.NUM_CLKS 7 CONFIG.MC_BOARD_INTRF_EN true CONFIG.CH0_LPDDR4_0_BOARD_INTERFACE ch0_lpddr4_trip1 CONFIG.CH1_LPDDR4_0_BOARD_INTERFACE ch1_lpddr4_trip1 CONFIG.sys_clk0_BOARD_INTERFACE lpddr4_clk1] $noc',
  'set fabric [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 fabric]',
  'set_property -dict [list CONFIG.NUM_SI 3 CONFIG.NUM_MI 1 CONFIG.NUM_CLKS 1] $fabric',
  'set host [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 host]',
  'set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1 CONFIG.NUM_CLKS 1] $host',
  f'set core [create_bd_cell -type ip -vlnv {vlnv} core]',
  'connect_bd_intf_net [get_bd_intf_pins cips/M_AXI_FPD] [get_bd_intf_pins host/S00_AXI]',
  'connect_bd_intf_net [get_bd_intf_pins host/M00_AXI] [get_bd_intf_pins core/s_axi_control]']
 for i,name in enumerate(required[:3]):lines.append(f'connect_bd_intf_net [get_bd_intf_pins core/{name}] [get_bd_intf_pins fabric/S{i:02d}_AXI]')
 lines+=['connect_bd_intf_net [get_bd_intf_pins fabric/M00_AXI] [get_bd_intf_pins noc/S00_AXI]',
  'set index 1','foreach path {FPD_CCI_NOC_0 FPD_CCI_NOC_1 FPD_CCI_NOC_2 FPD_CCI_NOC_3 LPD_AXI_NOC_0 PMC_NOC_AXI_0} {',
  'set source [get_bd_intf_pins cips/$path]','set sink [get_bd_intf_pins [format "noc/S%02d_AXI" $index]]',
  'connect_bd_intf_net $source $sink','set category [get_property CONFIG.MY_CATEGORY $source]',
  'set_property -dict [list CONFIG.CATEGORY $category CONFIG.CONNECTIONS {MC_0 {read_bw {1000} write_bw {1000}}}] $sink',
  'set clock [get_bd_pins noc/aclk$index]','set_property CONFIG.ASSOCIATED_BUSIF [get_property NAME $sink] $clock',
  'connect_bd_net [bd::clkrst::get_intf_driver_clk $source] $clock','incr index','}',
  'set_property CONFIG.ASSOCIATED_BUSIF S00_AXI [get_bd_pins noc/aclk0]',
  'set_property -dict [list CONFIG.CATEGORY pl CONFIG.CONNECTIONS {MC_0 {read_bw {1000} write_bw {1000}}}] [get_bd_intf_pins noc/S00_AXI]',
  'connect_bd_net [get_bd_pins cips/pl0_ref_clk] [get_bd_pins cips/m_axi_fpd_aclk] [get_bd_pins fabric/aclk] [get_bd_pins host/aclk] [get_bd_pins noc/aclk0] [get_bd_pins core/ap_clk]',
  'connect_bd_net [get_bd_pins cips/pl0_resetn] [get_bd_pins fabric/aresetn] [get_bd_pins host/aresetn] [get_bd_pins core/ap_rst_n]',
  'foreach pair {{CH0_LPDDR4_0 ch0_lpddr4_trip1} {CH1_LPDDR4_0 ch1_lpddr4_trip1} {sys_clk0 lpddr4_clk1}} {',
  'set pin [get_bd_intf_pins -quiet noc/[lindex $pair 0]]','if {[llength $pin]} {apply_bd_automation -rule xilinx.com:bd_rule:board -config [list Board_Interface [lindex $pair 1] Manual_Source Auto] $pin}','}',
  'assign_bd_address -offset 0xA4000000 -range 64K -target_address_space [get_bd_addr_spaces cips/M_AXI_FPD] [get_bd_addr_segs core/s_axi_control/Reg]',
  'assign_bd_address','validate_bd_design','save_bd_design',
  f'set f [open {quote(out/"address_segments.tsv")} w]',
  'foreach space [get_bd_addr_spaces] {foreach seg [get_bd_addr_segs -of_objects $space] {puts $f "$space\t$seg\t[get_property OFFSET $seg]\t[get_property RANGE $seg]"}}','close $f',
  'set core_spaces 0; set cpu_ddr 0',
  'foreach space [get_bd_addr_spaces] {',
  'if {[string match {/core/Data_m_axi_*} $space] || $space eq {/cips/FPD_CCI_NOC_0}} {',
  'set covered 0',
  'foreach seg [get_bd_addr_segs -of_objects $space] {if {[string match {*DDR_LOW0*} $seg]} {set base [get_property OFFSET $seg]; set span [get_property RANGE $seg]; if {$base <= 0x40000000 && ($base + $span) >= 0x44000000} {set covered 1}}}',
  'if {!$covered} {error "Shared DDR arena unavailable in $space"}',
  'if {[string match {/core/Data_m_axi_*} $space]} {incr core_spaces} else {set cpu_ddr 1}',
  '}',
  '}',
  'if {$core_spaces != 3 || !$cpu_ddr} {error "Expected three accelerator DDR masters and CPU DDR path"}',
  'generate_target all [get_files cnn_system.bd]','make_wrapper -files [get_files cnn_system.bd] -top -import',
  'set_property top cnn_system_wrapper [current_fileset]','update_compile_order -fileset sources_1',
  f'write_bd_tcl -force {quote(out/"actual_system_bd.tcl")}',f'write_hw_platform -fixed -force -file {quote(out/"preimplementation.xsa")}']
 if implement:
  lines += [f'foreach run [get_runs -filter {{IS_SYNTHESIS == 1}}] {{set_property STEPS.SYNTH_DESIGN.TCL.PRE {quote(hook)} $run}}',
   'launch_runs synth_1 -jobs 1','wait_on_run synth_1','if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "CNN system synthesis failed"}',
   'open_run synth_1',f'report_utilization -file {quote(out/"synth_utilization.rpt")}',f'report_timing_summary -file {quote(out/"synth_timing.rpt")}',
   'close_design','launch_runs impl_1 -to_step write_device_image -jobs 1','wait_on_run impl_1',
   'if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "CNN system implementation failed"}','open_run impl_1',
   f'report_utilization -file {quote(out/"utilization.rpt")}',f'report_timing_summary -file {quote(out/"timing.rpt")}',
   f'report_drc -file {quote(out/"drc.rpt")}',f'write_hw_platform -fixed -include_bit -force -file {quote(out/"implemented.xsa")}']
 (out/'system.tcl').write_text('\n'.join(lines+['exit'])+'\n');write_json(out/'prepared.json',{'hls':str(hls),'component_sha256':digest(component),'config_sha256':digest(config),'preset_sha256':digest(preset_path),'implement':implement,'evidence':'Prepared only; DDR allocation, synthesis and timing remain unmeasured','host_boundary':'A72 tokenized IDs/masks, complete document retained until final=1; all CNN arithmetic PL; deployed tokenization and transport separately charged'})

def resume(previous,out):
 previous=Path(previous).resolve();out=Path(out).resolve()
 state=json.loads((previous/'run.json').read_text())
 if state['state']!='failed':raise ValueError('Resume requires a stopped failed build')
 out.mkdir(exist_ok=False);archive=out/'failed_run_snapshots';archive.mkdir()
 project=previous/'project/kd_cnn_system.xpr';runs=previous/'project/kd_cnn_system.runs'
 if not project.exists():raise FileNotFoundError(project)
 for name in ['cnn_system_core_0_synth_1','synth_1','impl_1']:
  if (runs/name).exists():shutil.copytree(runs/name,archive/name)
 lines=['set_param general.maxThreads 2','set_param synth.maxThreads 1',f'open_project {quote(project)}',
  'reset_run cnn_system_core_0_synth_1','reset_run synth_1','reset_run impl_1',
  'launch_runs synth_1 -jobs 1','wait_on_run synth_1','if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Resumed synthesis failed"}',
  'open_run synth_1',f'report_utilization -file {quote(out/"synth_utilization.rpt")}',f'report_timing_summary -file {quote(out/"synth_timing.rpt")}',
  'close_design','launch_runs impl_1 -to_step write_device_image -jobs 1','wait_on_run impl_1',
  'if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Resumed implementation failed"}','open_run impl_1',
  f'report_utilization -file {quote(out/"utilization.rpt")}',f'report_timing_summary -file {quote(out/"timing.rpt")}',f'report_drc -file {quote(out/"drc.rpt")}',
  f'write_hw_platform -fixed -include_bit -force -file {quote(out/"implemented.xsa")}','exit']
 (out/'system.tcl').write_text('\n'.join(lines)+'\n')
 write_json(out/'prepared.json',{'resume_from':str(previous),'prior_run_sha256':digest(previous/'run.json'),'project_before_resume_sha256':digest(project),'preserved_failed_runs':str(archive),'boundary':'Reuse validated BD and completed interconnect synthesis; repeat interrupted core/top only, no kernel change'})


def execute(out,seconds):
 if os.name!='posix':raise RuntimeError('Execute from WSL')
 sys.path.insert(0,str(ROOT/'scripts'));from r1_challengers import resource_check
 out=Path(out).resolve();pre=resource_check(True);record=out/'run.json'
 if record.exists():raise FileExistsError('Retain prior attempt')
 if not 1<=seconds<=7200:raise ValueError('Bound must be <=7200 seconds')
 args=['/home/jp19126/Xilinx/2025.2/Vivado/bin/vivado','-mode','batch','-source',str(out/'system.tcl'),'-log',str(out/'vivado.log'),'-journal',str(out/'vivado.jou')]
 r={'command':args,'preflight':pre,'start_utc':datetime.now(timezone.utc).isoformat(),'state':'running'};start=time.monotonic();proc=None
 try:
  with (out/'stdout.log').open('x') as log:
   proc=subprocess.Popen(args,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);r['pid']=proc.pid;write_json(record,r)
   while proc.poll() is None:
    if (out/'STOP_REQUESTED').exists():raise InterruptedError('Coordinator stop requested; preserve vendor checkpoints')
    if time.monotonic()-start>seconds:raise TimeoutError('System build ceiling')
    obs=resource_check()
    with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(obs)+'\n')
    time.sleep(5)
   r.update(returncode=proc.returncode,state='completed' if proc.returncode==0 else 'failed')
 except BaseException as error:
  r.update(state='failed',error=repr(error));raise
 finally:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
  r['seconds']=time.monotonic()-start;write_json(record,r)
 if proc.returncode:raise RuntimeError('Vivado failed; inspect log')
def main():
 p=argparse.ArgumentParser();p.add_argument('--hls',type=Path);p.add_argument('--resume-from',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--implement',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--seconds',type=int,default=7200)
 a=p.parse_args()
 if not (a.out/'prepared.json').exists():
  if a.resume_from:resume(a.resume_from,a.out)
  else:prepare(a.hls,a.out,a.config,a.implement)
 if a.execute:execute(a.out,a.seconds)
if __name__=='__main__':main()
