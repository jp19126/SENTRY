# Invoked only by scripts/build_linear_hls.py after real configuration is complete.
# No default part, clock, interface or physical profile is hidden in this file.
if {![info exists ::env(GATE_HLS_SETTINGS)]} {
    error "GATE_HLS_SETTINGS must name settings generated from configs/project.json"
}
source $::env(GATE_HLS_SETTINGS)
open_project $gate_project
set_top gate_linear_top
add_files $gate_top -cflags $gate_cflags
add_files -tb $gate_testbench -cflags $gate_cflags
open_solution solution -flow_target $gate_flow_target
set_part $gate_part
create_clock -period $gate_clock_ns -name default
set_clock_uncertainty $gate_clock_uncertainty_ns
source $gate_interface_tcl
if {$gate_action eq "cosim"} {
    # Setup generates the verification harness; it is not a passing RTL simulation.
    cosim_design -setup -rtl verilog -tool xsim -trace_level none -argv $gate_cosim_argv
    set gate_sim_dir [file join $gate_project solution sim verilog]
    set gate_cosim_helper [file normalize [file join [file dirname [info script]] .. scripts prepare_hls_cosim.py]]
    puts [exec python3 $gate_cosim_helper --sim-dir $gate_sim_dir]
    set gate_previous_directory [pwd]
    cd $gate_sim_dir
    set gate_sim_result [catch {source [file join $gate_sim_dir run_sim.tcl]} gate_sim_error gate_sim_options]
    cd $gate_previous_directory
    if {$gate_sim_result != 0} {
        return -options $gate_sim_options $gate_sim_error
    }
    puts [exec python3 $gate_cosim_helper --sim-dir $gate_sim_dir --verify]
} else {
    csim_design
    if {$gate_action eq "synthesize"} {
        csynth_design
    }
}
exit

