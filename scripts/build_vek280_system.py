"""Generate the study-specific real-service Vivado system wrapper and platform build.

No board connection/programming. --execute invokes the selected offline phase.
"""
from pathlib import Path
import argparse,json,os,re,shutil,signal,subprocess,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
# Actual CIPS clock from corrected selected-board prepare20261008T003902_321496Z.
# Nominal target200MHz with installed33.3333MHz reference; no physical observation.
ACTUAL_PL_CLOCK_HZ=199999817

def emit_core(out,length=256):
    seq=(ROOT/'hardware/gate_detector_sequencer.sv').read_text()
    public=seq[seq.index(' input wire clk'):seq.index(' // Single AXI-Lite master;')].rstrip().rstrip(',')
    public=public.replace('output reg','output wire')
    external=set(re.findall(r'\b(?:config_\w+|window_\w+|result_\w+|clk|resetn)\b',public))
    control=(ROOT/'hardware/gate_detector_control_tb.sv').read_text()
    declarations=control[control.index(' logic  clk;'):control.index(' logic [2:0] fault_mode;')]
    declarations='\n'.join(line for line in declarations.splitlines() if not any(re.search(r'\b'+name+r'\b',line) for name in external) and not re.search(r'\bm_axi_',line))
    # Exact existing control demux and real mover connection.
    connected=control[control.index(' gate_detector_sequencer dut(.*);'):control.index(' gate_mover_axi_memory #')]
    ports=[public]
    top_names=set(external)
    axi=[('awvalid','output',1),('awready','input',1),('awid','output',1),('awaddr','output',64),('awlen','output',8),('awsize','output',3),('awburst','output',2),('awlock','output',2),('wvalid','output',1),('wready','input',1),('wdata','output',32),('wstrb','output',4),('wlast','output',1),('wid','output',1),('bvalid','input',1),('bready','output',1),('bresp','input',2),('bid','input',1),('arvalid','output',1),('arready','input',1),('arid','output',1),('araddr','output',64),('arlen','output',8),('arsize','output',3),('arburst','output',2),('arlock','output',2),('rvalid','input',1),('rready','output',1),('rdata','input',32),('rlast','input',1),('rid','input',1),('rresp','input',2)]
    def port(name,direction,width,interface,signal,protocol):
        top_names.add(name)
        parameter=''
        if signal=='awvalid':parameter=f'(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME {interface}, PROTOCOL {protocol}, DATA_WIDTH 32, ADDR_WIDTH 64, ID_WIDTH 1, FREQ_HZ 200000000, MAX_BURST_LENGTH 16, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *)\n '
        shape='' if width==1 else f'[{width-1}:0] '
        ports.append(parameter+f'(* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 {interface} {signal.upper()}" *) {direction} wire {shape}{name}')
    module_body=declarations+'''
 wire [1:0] slave_awready,slave_wready,slave_bvalid,slave_arready,slave_rvalid;
 wire [1:0] slave_bresp[0:1],slave_rresp[0:1];
 wire [31:0] slave_rdata[0:1];
 assign ctl_awready=slave_awready[ctl_target];assign ctl_wready=slave_wready[ctl_target];
 assign ctl_bvalid=slave_bvalid[ctl_target];assign ctl_bresp=slave_bresp[ctl_target];
 assign ctl_arready=slave_arready[ctl_target];assign ctl_rvalid=slave_rvalid[ctl_target];
 assign ctl_rresp=slave_rresp[ctl_target];assign ctl_rdata=slave_rdata[ctl_target];
'''+connected
    interfaces=[]
    for index,source,module,names in [(0,'linear_axi_tb.sv','gate_linear_top',['a','w','o']),(1,'fixed_fp32_axi_tb.sv','gate_fixed_fp32_top',['x','y','z','accumulators','output','codes'])]:
        text=(ROOT/'hardware'/source).read_text();connection=re.search(r'    '+module+r' dut \((.*?)\n    \);',text,re.S).group(1)
        for name in names:
            interface=f'M_AXI_E{index}_{name.upper()}';prefix=f'm_axi_e{index}_{name}_';interfaces.append(interface)
            for signal,direction,width in axi:port(prefix+signal,direction,width,interface,signal,'AXI3')
            connection=re.sub(r'\b'+name+r'_bus\.([a-z]+)',lambda m:prefix+m[1],connection)
        for name in ['awready','wready','bvalid','bresp','arready','rvalid','rdata','rresp']:connection=connection.replace('c_'+name,f'slave_{name}[{index}]')
        for name in ['awvalid','wvalid','bready','arvalid','rready']:connection=connection.replace('c_'+name,f'ctl_{name} && ctl_target=={index}')
        for name in ['awaddr','araddr','wdata','wstrb']:connection=connection.replace('c_'+name,'ctl_'+name+('[6:0]' if index==0 and name in ['awaddr','araddr'] else ''))
        connection=connection.replace('.interrupt(interrupt)',f'.interrupt(engine_interrupt[{index}])').replace('.interrupt()',f'.interrupt(engine_interrupt[{index}])')
        module_body+=' '+module+f' arithmetic_{index} ('+connection+'\n );\n'
    mover_signals=set(re.findall(r'\bm_axi_([a-z]+)\b',connected))
    for signal,direction,width in axi:
        if signal in mover_signals:port('m_axi_'+signal,direction,width,'M_AXI_MOVER',signal,'AXI4')
    interfaces.append('M_AXI_MOVER')
    clock_interfaces=':'.join(interfaces)
    header=f'// Generated by scripts/build_vek280_system.py from actual validated port maps.\nmodule gate_detector_core #(parameter integer TOKENS={length}) (\n'
    text=header+',\n'.join(ports)+'\n);\n'+module_body+'\nendmodule\n'
    text=text.replace(' input wire clk, resetn,',f' (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER = "XIL_INTERFACENAME clk, FREQ_HZ 200000000, ASSOCIATED_BUSIF {clock_interfaces}, ASSOCIATED_RESET resetn" *) input wire clk,\n (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 resetn RST", X_INTERFACE_PARAMETER = "XIL_INTERFACENAME resetn, POLARITY ACTIVE_LOW" *) input wire resetn,')
    (out/'gate_detector_core.sv').write_text(text.replace('module gate_detector_core #','module gate_detector_core_impl #').replace('200000000',str(ACTUAL_PL_CLOCK_HZ)))
    # IP Integrator accepts Verilog module-reference tops, with SV children.
    wrapper=header+',\n'.join(ports)+'\n);\n gate_detector_core_impl #(.TOKENS(TOKENS)) impl (\n'+',\n'.join(' .'+name+'('+name+')' for name in sorted(top_names))+'\n );\nendmodule\n'
    wrapper=wrapper.replace(' input wire clk, resetn,',f' (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER="XIL_INTERFACENAME clk, FREQ_HZ 200000000, ASSOCIATED_BUSIF {clock_interfaces}, ASSOCIATED_RESET resetn" *) input wire clk,\n (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 resetn RST", X_INTERFACE_PARAMETER="XIL_INTERFACENAME resetn, POLARITY ACTIVE_LOW" *) input wire resetn,')
    (out/'gate_detector_core.v').write_text(wrapper.replace('200000000',str(ACTUAL_PL_CLOCK_HZ)))
    mailbox=(ROOT/'hardware/gate_host_mailbox.sv').read_text()
    mailbox_ports=mailbox.split(') (',1)[1].split(');',1)[0].replace('output reg','output wire')
    mailbox_names=external|set(re.findall(r'\bs_axi_\w+',mailbox_ports))
    mailbox_wrapper=f'module gate_host_mailbox_bd #(parameter integer TOKENS={length}) ('+mailbox_ports+');\n gate_host_mailbox #(.TOKENS(TOKENS)) impl (\n'+',\n'.join(' .'+name+'('+name+')' for name in sorted(mailbox_names))+'\n );\nendmodule\n'
    (out/'gate_host_mailbox_bd.v').write_text(mailbox_wrapper.replace('200000000',str(ACTUAL_PL_CLOCK_HZ)))
    return sorted(external-{'clk','resetn'}),interfaces

def emit_platform(config,out,public,interfaces,phase,length=256,lanes=64,ip_cache=None):
    import xml.etree.ElementTree as ET
    from run_fixed_fp32_rtl import reachable_rtl,tcl_literal
    from plan_detector_commands import create_plan,write_rtl_plan
    plan=create_plan(config,length);write_rtl_plan(plan,out)
    (out/'command_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    preset_path=Path(config['hardware']['board_definition']['board_definition_path']).with_name('preset.xml')
    tree=ET.parse(preset_path).getroot()
    xml=tree.find(".//ip_preset[@preset_proc_name='ps_pmc_fixed_io_preset']//user_hier_parameter[@name='CONFIG.PS_PMC_CONFIG']")
    def xml_dict(node):
        return {child.attrib['name']:xml_dict(child) if list(child) else child.attrib['value'] for child in node}
    preset=xml_dict(xml)
    preset.update(DESIGN_MODE='1',PS_USE_PMCPL_CLK0='1',PMC_CRP_PL0_REF_CTRL_FREQMHZ='200',PS_USE_M_AXI_FPD='1',PS_NUM_FABRIC_RESETS='1',PS_USE_FPD_CCI_NOC='1',PS_USE_NOC_LPD_AXI0='1',PMC_USE_PMC_NOC_AXI0='1')
    def nested_pairs(value):
        # CIPS hierarchical settings use a pair for one child and a list of
        # pairs for multiple children (as retained in the actual saved BD).
        result=[]
        for key,item in value.items():
            inside='{'+nested_pairs(item)+'}' if isinstance(item,dict) else str(item)
            pair=str(key)+' '+inside
            result.append(pair if len(value)==1 else '{'+pair+'}')
        return ' '.join(result)
    def tcl_dict(value):
        return ' '.join('{'+str(k)+'} {'+(nested_pairs(v) if isinstance(v,dict) else str(v))+'}' for k,v in value.items())
    fp=ROOT/'build/fixed_fp32_hls/project/solution/syn/verilog'
    variant=config['hardware']['hls']['cached_linear']['profile_kernel_variants'][f'lanes{lanes}']
    parent='linear_hls_cached_static_bank' if variant=='weight_cache_static_bank' else 'linear_hls_cached'
    integer=ROOT/'build'/parent/f'lanes{lanes}'/'project/solution/syn/verilog'
    if not (integer/'gate_linear_top.v').is_file():raise ValueError('Completed selected integer RTL is missing')
    fpfiles,ips=reachable_rtl(fp)
    ipfolder=out/'arithmetic_ip';ipfolder.mkdir(exist_ok=True)
    copied=[]
    for source in ips:
        dest=ipfolder/source.name
        # Preserve arithmetic IP parameters; this deployment only needs synthesis products.
        dest.write_text(source.read_text().replace('generate_target {synthesis simulation}','generate_target {synthesis}'))
        copied.append(dest)
    sourcefiles=[out/'gate_detector_core.v',out/'gate_detector_core.sv',out/'gate_host_mailbox_bd.v',ROOT/'hardware/gate_host_mailbox.sv',ROOT/'hardware/gate_detector_sequencer.sv',ROOT/'hardware/gate_data_mover.sv']+sorted(integer.glob('*.v'))+fpfiles
    memfiles=[out/'command_plan.mem',out/'command_plan_constants.svh']+sorted(fp.glob('*.dat'))
    # Freeze the exact RTL/data compiled by this run, including parameterized control.
    def freeze(path):
        if path.is_relative_to(out):return path
        dest=out/'sources'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);return dest
    sourcefiles=[freeze(path) for path in sourcefiles]
    memfiles=[freeze(path) for path in memfiles]
    limits=config['execution_control']['board_free_continuation']['resource_partition']
    jobs=limits['vendor_jobs'];threads=limits['vendor_max_threads']
    if jobs!=1 or threads!=2:raise ValueError('Current cause-identified integration budget is jobs1/maxThreads2')
    hook=out/'limit_threads.tcl';hook.write_text('set_param general.maxThreads '+str(threads)+'\nset_param synth.maxThreads 1\nputs "SENTRY_ACTUAL_MAX_THREADS [get_param general.maxThreads]"\nputs "SENTRY_ACTUAL_SYNTH_MAX_THREADS [get_param synth.maxThreads]"\nif {[get_param synth.maxThreads] != 1} {error "Inner synthesis worker cap not applied"}\n')
    lines=[f'set_param general.maxThreads {threads}','set_param synth.maxThreads 1','create_project -force sentry_vek280 '+tcl_literal(out/'project')+' -part '+config['hardware']['part'],
        'set_property board_part '+config['hardware']['board_definition']['board_part']+' [current_project]',
        'set_property target_language Verilog [current_project]',
        'set_property include_dirs '+tcl_literal(out)+' [current_fileset]']
    if ip_cache is not None:
        lines += ['config_ip_cache -use_cache_location '+tcl_literal(ip_cache),
            'set_property IP_CACHE_PERMISSIONS {read} [current_project]',
            'puts \"SENTRY_IP_CACHE_READ_ONLY [get_property IP_OUTPUT_REPO [current_project]]\"']
    lines+=['source '+tcl_literal(file) for file in copied]
    lines+=['add_files -norecurse [list '+' '.join(tcl_literal(file) for file in sourcefiles+memfiles)+']',
        'create_bd_design sentry_system',
        'set cips [create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips:3.4 cips]',
        'set_property CONFIG.PS_PMC_CONFIG {'+tcl_dict(preset)+'} $cips',
        'set actual_ps_config [get_property CONFIG.PS_PMC_CONFIG $cips]',
        'puts "SENTRY_ACTUAL_CIPS_CONFIG $actual_ps_config"',
        'foreach peripheral {PS_UART0_PERIPHERAL PMC_SD1_PERIPHERAL} {',
        ' set enabled 0',
        ' foreach pair [dict get $actual_ps_config $peripheral] {if {[lindex $pair 0] eq "ENABLE"} {set enabled [lindex $pair 1]}}',
        ' if {$enabled ne "1"} {error "Selected board peripheral $peripheral is not enabled"}',
        '}',
        'puts "SENTRY_ACTUAL_PL_CLOCK [get_property CONFIG.FREQ_HZ [get_bd_pins cips/pl0_ref_clk]]"',
        'puts "SENTRY_CONFIGURED_CIPS_PINS [get_bd_intf_pins -of_objects $cips]"',
        'puts "SENTRY_CONFIGURED_CIPS_SIGNALS [get_bd_pins -of_objects $cips]"',
        'set noc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 noc]',
        'set_property -dict [list CONFIG.NUM_SI 7 CONFIG.NUM_MI 0 CONFIG.NUM_MC 1 CONFIG.NUM_CLKS 7 CONFIG.MC_BOARD_INTRF_EN true CONFIG.CH0_LPDDR4_0_BOARD_INTERFACE ch0_lpddr4_trip1 CONFIG.CH1_LPDDR4_0_BOARD_INTERFACE ch1_lpddr4_trip1 CONFIG.sys_clk0_BOARD_INTERFACE lpddr4_clk1] $noc',
        'puts "SENTRY_CONFIGURED_NOC_PINS [get_bd_intf_pins -of_objects $noc]"',
        'set fabric [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 fabric]',
        'set_property -dict [list CONFIG.NUM_SI 10 CONFIG.NUM_MI 1 CONFIG.NUM_CLKS 1] $fabric',
        'set host_fabric [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 host_fabric]',
        'set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1 CONFIG.NUM_CLKS 1] $host_fabric',
        'set core [create_bd_cell -type module -reference gate_detector_core core]',
        'set mailbox [create_bd_cell -type module -reference gate_host_mailbox_bd mailbox]',
        f'set_property CONFIG.TOKENS {length} $mailbox',
        'connect_bd_intf_net [get_bd_intf_pins cips/M_AXI_FPD] [get_bd_intf_pins host_fabric/S00_AXI]',
        'connect_bd_intf_net [get_bd_intf_pins host_fabric/M00_AXI] [get_bd_intf_pins mailbox/S_AXI]']
    for i,interface in enumerate(interfaces):
        lines.append(f'connect_bd_intf_net [get_bd_intf_pins core/{interface}] [get_bd_intf_pins fabric/S{i:02d}_AXI]')
    lines += ['connect_bd_intf_net [get_bd_intf_pins fabric/M00_AXI] [get_bd_intf_pins noc/S00_AXI]',
        'set ps_paths {FPD_CCI_NOC_0 FPD_CCI_NOC_1 FPD_CCI_NOC_2 FPD_CCI_NOC_3 LPD_AXI_NOC_0 PMC_NOC_AXI_0}',
        'set index 1',
        'foreach path $ps_paths {',
        ' set source [get_bd_intf_pins cips/$path]',
        ' set sink [get_bd_intf_pins [format "noc/S%02d_AXI" $index]]',
        ' connect_bd_intf_net $source $sink',
        ' set category [get_property CONFIG.MY_CATEGORY $source]',
        ' set_property -dict [list CONFIG.CATEGORY $category CONFIG.CONNECTIONS {MC_0 {read_bw {1000} write_bw {1000}}}] $sink',
        ' set clock [get_bd_pins noc/aclk$index]',
        ' set_property CONFIG.ASSOCIATED_BUSIF [get_property NAME $sink] $clock',
        ' connect_bd_net [bd::clkrst::get_intf_driver_clk $source] $clock',
        ' incr index',
        '}',
        'set_property CONFIG.ASSOCIATED_BUSIF S00_AXI [get_bd_pins noc/aclk0]',
        'set_property -dict [list CONFIG.CATEGORY pl CONFIG.CONNECTIONS {MC_0 {read_bw {1000} write_bw {1000}}}] [get_bd_intf_pins noc/S00_AXI]',
        'connect_bd_net [get_bd_pins cips/pl0_ref_clk] [get_bd_pins cips/m_axi_fpd_aclk] [get_bd_pins fabric/aclk] [get_bd_pins host_fabric/aclk] [get_bd_pins noc/aclk0] [get_bd_pins core/clk] [get_bd_pins mailbox/clk]',
        'connect_bd_net [get_bd_pins cips/pl0_resetn] [get_bd_pins fabric/aresetn] [get_bd_pins host_fabric/aresetn] [get_bd_pins core/resetn] [get_bd_pins mailbox/resetn]']
    for signal in public:lines.append(f'connect_bd_net [get_bd_pins core/{signal}] [get_bd_pins mailbox/{signal}]')
    lines += ['foreach pair {{CH0_LPDDR4_0 ch0_lpddr4_trip1} {CH1_LPDDR4_0 ch1_lpddr4_trip1} {sys_clk0 lpddr4_clk1}} {',
        ' set pin [get_bd_intf_pins -quiet noc/[lindex $pair 0]]',
        ' if {[llength $pin]} { apply_bd_automation -rule xilinx.com:bd_rule:board -config [list Board_Interface [lindex $pair 1] Manual_Source Auto] $pin }',
        '}',
        'assign_bd_address -offset 0xA4000000 -range 4K -target_address_space [get_bd_addr_spaces cips/M_AXI_FPD] [get_bd_addr_segs mailbox/S_AXI/reg0]',
        'assign_bd_address',
        'save_bd_design',
        'set address_file [open '+tcl_literal(out/'address_segments.tsv')+' w]',
        'foreach space [get_bd_addr_spaces] { foreach seg [get_bd_addr_segs -of_objects $space] {',
        ' puts $address_file "$space\\t$seg\\t[get_property OFFSET $seg]\\t[get_property RANGE $seg]"',
        '}}',
        'close $address_file',
        'validate_bd_design',
        'generate_target all [get_files sentry_system.bd]',
        'make_wrapper -files [get_files sentry_system.bd] -top -import',
        'set_property top sentry_system_wrapper [current_fileset]',
        'update_compile_order -fileset sources_1',
        'write_bd_tcl -force '+tcl_literal(out/'actual_system_bd.tcl'),
        'write_hw_platform -fixed -force -file '+tcl_literal(out/'sentry_vek280_preimplementation.xsa')]
    if ip_cache is not None:
        lines += ['set cache_file [open '+tcl_literal(out/'ip_cache_matches.tsv')+' w]',
            'foreach ip [get_ips] {puts $cache_file \"$ip\\t[config_ip_cache -cache_has_match $ip]\\t[config_ip_cache -cache_was_used $ip]\"}',
            'close $cache_file',
            'config_ip_cache -report -rptfile '+tcl_literal(out/'ip_cache_report.rpt')]
    if phase=='implement':
        lines += ['foreach run [get_runs -filter {IS_SYNTHESIS == 1}] {set_property STEPS.SYNTH_DESIGN.TCL.PRE '+tcl_literal(hook)+' $run}',
            'foreach property [list_property [get_runs impl_1]] {if {[string match "STEPS.*.TCL.PRE" $property]} {set_property $property '+tcl_literal(hook)+' [get_runs impl_1]}}',
            f'launch_runs synth_1 -jobs {jobs}','wait_on_run synth_1',
            'if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Integrated synthesis failed"}']
        if ip_cache is not None:
            lines += ['set cache_file [open '+tcl_literal(out/'ip_cache_final_use.tsv')+' w]',
                'foreach ip [get_ips] {puts $cache_file "$ip\\t[config_ip_cache -cache_has_match $ip]\\t[config_ip_cache -cache_was_used $ip]"}',
                'close $cache_file','config_ip_cache -report -rptfile '+tcl_literal(out/'ip_cache_final_report.rpt')]
        lines += [f'launch_runs impl_1 -to_step write_device_image -jobs {jobs}','wait_on_run impl_1',
            'if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Integrated implementation failed"}',
            'open_run impl_1','report_utilization -file '+tcl_literal(out/'integrated_utilization.rpt'),
            'report_timing_summary -file '+tcl_literal(out/'integrated_timing.rpt'),
            'report_drc -file '+tcl_literal(out/'integrated_drc.rpt'),
            'write_hw_platform -fixed -include_bit -force -file '+tcl_literal(out/'sentry_vek280.xsa')]
    lines += ['close_project','exit']
    path=out/'build_system.tcl';path.write_text('\n'.join(lines)+'\n')
    return path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--phase',choices=['prepare','implement'],default='prepare')
    ap.add_argument('--length',type=int,choices=[128,256,512],default=256)
    ap.add_argument('--lanes',type=int,choices=[16,32,64],default=64)
    ap.add_argument('--limit-seconds',type=int,default=900)
    ap.add_argument('--ip-cache',type=Path,help='Existing Vivado IP cache, reused read-only when vendor configuration/version matching permits')
    args=ap.parse_args()
    if not 1<=args.limit_seconds<=7200:raise ValueError('Bound must be 1..7200 seconds')
    config=json.loads((ROOT/'configs/project.json').read_text(encoding='utf-8-sig'))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=ROOT/'build/vek280_system'/f'l{args.length}'/f'lanes{args.lanes}'/(stamp+'_'+args.phase)
    out.mkdir(parents=True,exist_ok=False)
    ip_cache=None
    if args.ip_cache is not None:
        ip_cache=(args.ip_cache if args.ip_cache.is_absolute() else ROOT/args.ip_cache).resolve()
        if not ip_cache.is_relative_to((ROOT/'build/vek280_system').resolve()) or not ip_cache.is_dir():
            raise ValueError('Existing cache must be inside the actual system build tree')
    public,interfaces=emit_core(out,args.length)
    (out/'core_interfaces.json').write_text(json.dumps(dict(mailbox_signals=public,memory_interfaces=interfaces),indent=2)+'\n')
    tcl=emit_platform(config,out,public,interfaces,args.phase,args.length,args.lanes,ip_cache)
    command=[config['hardware']['hls']['vivado_executable'],'-mode','batch','-source',str(tcl),'-log',str(out/'vendor.log'),'-journal',str(out/'vendor.jou')]
    if args.phase=='implement':command=['taskset','-c','0-5']+command
    record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),phase=args.phase,length=args.length,lanes=args.lanes,integer_kernel_variant=config['hardware']['hls']['cached_linear']['profile_kernel_variants'][f'lanes{args.lanes}'],state='prepared',passed=False,ip_cache_source=str(ip_cache) if ip_cache else None,ip_cache_mode='vendor exact-match read-only' if ip_cache else 'project default',command=command,limit_seconds=args.limit_seconds,source_snapshot=str(out/'sources'),plan=str(out/'command_plan.json'),board_access=False,physical_measurement=False)
    def save():
        (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copyfile(Path(__file__),out/'executed_driver.py')
    save();print(json.dumps(dict(output=str(out),**record),indent=2),flush=True)
    if not args.execute:return
    if os.name=='nt':raise RuntimeError('Run vendor tools with WSL Linux Python')
    pause=ROOT/'results/goal2/pause.request'
    proc=None
    try:
        if pause.exists():raise RuntimeError('Pause marker present')
        mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapFree:','MemFree:','Cached:','SReclaimable:','Dirty:','Writeback:'))}
        record['resource_observation']={'cpu_count':os.cpu_count(),'parent_affinity':sorted(os.sched_getaffinity(0)),'planned_vendor_affinity':[0,1,2,3,4,5] if args.phase=='implement' else sorted(os.sched_getaffinity(0)),'jobs':1,'max_threads':2,'synthesis_workers':1,'runtime_wsl_floor_kib':4*1024*1024,'runtime_windows_floor_kib':2*1024*1024,'memory_kib':mem,'observed_utc':datetime.now(timezone.utc).isoformat()}
        if args.phase=='implement':
            host=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],text=True,capture_output=True,timeout=30)
            if host.returncode:raise RuntimeError('Cannot observe Windows memory headroom')
            record['resource_observation']['windows_free_kib']=int(host.stdout.strip())
            if not set(range(6))<=os.sched_getaffinity(0) or mem['MemAvailable']<8*1024*1024 or record['resource_observation']['windows_free_kib']<4*1024*1024:raise RuntimeError('Implementation needs CPUs0..5,8GiB WSL and4GiB Windows available memory')
        record['state']='running';save()
        with (out/'vendor.stdout.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['vendor_pid']=proc.pid;save()
            began=time.monotonic();last_observe=began
            while proc.poll() is None:
                if pause.exists() or time.monotonic()-began>args.limit_seconds:
                    raise RuntimeError('Pause marker present' if pause.exists() else 'Bounded execution limit reached')
                if 'actual_vendor_affinity' not in record and time.monotonic()-began>.5:
                    record['actual_vendor_affinity']=sorted(os.sched_getaffinity(proc.pid));save()
                if time.monotonic()-last_observe>5:
                 current={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith(('MemAvailable:','SwapFree:','MemFree:','Cached:','SReclaimable:','Dirty:','Writeback:'))}
                 observed_host=subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],text=True,capture_output=True,timeout=10)
                 if observed_host.returncode:raise RuntimeError('Runtime resource observation failed; preserve assigned vendor work')
                 current['windows_free_kib']=int(observed_host.stdout.strip())
                 process_result=subprocess.run(['ps','-eo','pid,ppid,pgid,stat,rss,comm'],capture_output=True,text=True,timeout=5,check=True)
                 assigned=[]
                 for line in process_result.stdout.splitlines()[1:]:
                  fields=line.split()
                  if len(fields)==6 and int(fields[2])==proc.pid:assigned.append(dict(pid=int(fields[0]),ppid=int(fields[1]),pgid=int(fields[2]),state=fields[3],rss_kib=int(fields[4]),command_name=fields[5]))
                 observation=dict(observed_utc=datetime.now(timezone.utc).isoformat(),**current,assigned_vendor_processes=assigned)
                 with (out/'resource_observations.jsonl').open('a',encoding='utf-8') as observations:observations.write(json.dumps(observation)+'\n')
                 record['latest_memory_kib']={k:v for k,v in observation.items() if k!='assigned_vendor_processes'};save();last_observe=time.monotonic()
                 if current['MemAvailable']<4*1024*1024 or current['windows_free_kib']<2*1024*1024:raise RuntimeError('Resource guard: runtime needs4GiB WSL and2GiB Windows available memory')
                time.sleep(.5)
        record['returncode']=proc.returncode
        if proc.returncode:raise RuntimeError('Vivado failed; inspect vendor.stdout.log')
        record.update(state='returned',passed=True)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        record.update(state='stopped' if proc is not None and proc.returncode and proc.returncode<0 else 'failed',error=str(exc))
        raise
    finally:save()
if __name__=='__main__':main()
