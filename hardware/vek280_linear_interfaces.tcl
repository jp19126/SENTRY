# VEK280 PL IP interface selected for offline Goal 4 characterization.
# Distinct AXI ports do not imply independent physical DDR channels or overlap.
set_directive_interface -mode m_axi -offset slave -bundle gmem_a -depth 262144 "gate_linear_top" activations
set_directive_interface -mode m_axi -offset slave -bundle gmem_w -depth 262144 "gate_linear_top" weights
set_directive_interface -mode m_axi -offset slave -bundle gmem_o -depth 262144 "gate_linear_top" output
foreach arg {activations weights output rows inner outputs weight_bits status return} {
    set_directive_interface -mode s_axilite -bundle control "gate_linear_top" $arg
}
