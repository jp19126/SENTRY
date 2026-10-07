// Passive lanes4 diagnostic, compiled as a second top beside linear_axi_tb.
// No DUT/TB drives, delays, force/release, waveforms or growing histories.
// State occupancy covers accepted top start through (excluding) sampled done.
// Stage duration covers sampled ap_start rise through sampled ap_done.
// Exact correctness and termination remain in the unchanged linear_axi_tb.
`timescale 1ns/1ps
`define DE linear_axi_tb.dut.grp_linear_engine_4_4_16_64_256_s_fu_137
`define DI `DE.grp_linear_engine_4_4_16_64_256_Pipeline_VITIS_LOOP_65_3_VITIS_LOOP_66_4_fu_246
`define DA `DE.grp_linear_engine_4_4_16_64_256_Pipeline_VITIS_LOOP_74_7_VITIS_LOOP_75_8_fu_254
`define DW `DE.grp_linear_engine_4_4_16_64_256_Pipeline_VITIS_LOOP_82_9_VITIS_LOOP_83_10_fu_271
`define DC `DE.grp_linear_engine_Pipeline_VITIS_LOOP_91_11_VITIS_LOOP_92_12_VITIS_LOOP_93_13_VITIS_s_fu_305
`define DS `DE.grp_linear_engine_4_4_16_64_256_Pipeline_VITIS_LOOP_125_16_VITIS_LOOP_126_17_fu_288
module linear_diag;
    wire [4:0] stage_start={`DS.ap_start,`DC.ap_start,`DW.ap_start,`DA.ap_start,`DI.ap_start};
    wire [4:0] stage_done={`DS.ap_done,`DC.ap_done,`DW.ap_done,`DA.ap_done,`DI.ap_done};
    // Initialization has no stall condition; compute's generated block is constant0.
    wire [4:0] stage_block={`DS.ap_block_pp0_stage0_subdone,`DC.ap_block_pp0_stage0_subdone,
        `DW.ap_block_pp0_stage0_subdone,`DA.ap_block_pp0_stage0_subdone,1'b0};
    wire [9:0] engine_state=`DE.ap_CS_fsm;
    wire top_start=linear_axi_tb.dut.ap_start && linear_axi_tb.dut.ap_CS_fsm_state1;
    wire top_done=linear_axi_tb.dut.ap_done;
    bit active=0;
    bit [4:0] stage_active=0, previous_start=0;
    longint unsigned cycle_count=0, call_start=0;
    integer call_index=0, diagnostic_errors=0;
    longint unsigned fsm_cycles[0:9];
    longint unsigned calls[0:4], finishes[0:4], starts[0:4];
    longint unsigned sums[0:4], minima[0:4], maxima[0:4], stalls[0:4];
    longint unsigned weight_ar,weight_r,store_aw,store_w,store_b;
    function automatic string stage_name(input integer index);
        case(index)
            0:stage_name="initialize"; 1:stage_name="activation_load";
            2:stage_name="weight_load"; 3:stage_name="compute";
            default:stage_name="store";
        endcase
    endfunction
    always @(posedge linear_axi_tb.clk) begin : observe
        longint unsigned duration,state_sum,R,O,I,expected_calls;
        if (!linear_axi_tb.resetn) begin
            cycle_count=0; active=0; previous_start=0; stage_active=0; call_index=0;
        end else begin
            cycle_count=cycle_count+1;
            if (top_start) begin
                active=1; call_start=cycle_count; diagnostic_errors=0; stage_active=0;
                weight_ar=0; weight_r=0; store_aw=0; store_w=0; store_b=0;
                for (integer j=0;j<10;j=j+1) fsm_cycles[j]=0;
                for (integer j=0;j<5;j=j+1) begin
                    calls[j]=0; finishes[j]=0; starts[j]=0; sums[j]=0;
                    minima[j]=64'hffffffffffffffff; maxima[j]=0; stalls[j]=0;
                end
            end
            if (active && !top_done) begin
                if (!$onehot(engine_state)) diagnostic_errors=diagnostic_errors+1;
                for (integer j=0;j<10;j=j+1)
                    if (engine_state[j]) fsm_cycles[j]=fsm_cycles[j]+1;
                for (integer j=0;j<5;j=j+1) begin
                    if (stage_start[j] && !previous_start[j]) begin
                        if (stage_active[j]) diagnostic_errors=diagnostic_errors+1;
                        stage_active[j]=1; starts[j]=cycle_count; calls[j]=calls[j]+1;
                    end
                    if (stage_active[j] && stage_done[j]) begin
                        duration=cycle_count-starts[j]; sums[j]=sums[j]+duration;
                        if (duration<minima[j]) minima[j]=duration;
                        if (duration>maxima[j]) maxima[j]=duration;
                        finishes[j]=finishes[j]+1; stage_active[j]=0;
                    end
                    if (stage_active[j] && stage_block[j]) stalls[j]=stalls[j]+1;
                end
                // Cause counters may overlap and must not be added as disjoint waits.
                if (stage_active[2]) begin
                    if (`DW.ap_enable_reg_pp0_iter1 && `DW.ap_block_state2_io) weight_ar=weight_ar+1;
                    if (`DW.ap_enable_reg_pp0_iter12 && `DW.ap_block_state13_pp0_stage0_iter12) weight_r=weight_r+1;
                end
                if (stage_active[4]) begin
                    if (`DS.ap_enable_reg_pp0_iter3 && `DS.ap_block_state4_io) store_aw=store_aw+1;
                    if (`DS.ap_enable_reg_pp0_iter4 && `DS.ap_block_state5_io_grp1
                        && !`DS.ap_block_pp0_stage0_subdone_grp1_done_reg) store_w=store_w+1;
                    if (`DS.ap_enable_reg_pp0_iter15 && `DS.ap_block_state16_pp0_stage0_iter15_grp2
                        && !`DS.ap_block_pp0_stage0_subdone_grp2_done_reg) store_b=store_b+1;
                end
            end
            if (active && top_done) begin
                R=(linear_axi_tb.rows+3)/4; O=linear_axi_tb.outputs/16; I=linear_axi_tb.inner/64;
                state_sum=0;
                for (integer j=0;j<10;j=j+1) state_sum=state_sum+fsm_cycles[j];
                if (state_sum!=cycle_count-call_start || stage_active!=0) diagnostic_errors=diagnostic_errors+1;
                for (integer j=0;j<5;j=j+1) begin
                    expected_calls=R*O;
                    if (j>=1 && j<=3) expected_calls=expected_calls*I;
                    if (calls[j]!=expected_calls || finishes[j]!=expected_calls) diagnostic_errors=diagnostic_errors+1;
                end
                // Array elements map exactly to generated engine states1 through10.
                $write("GATE_RTL_DIAGNOSTIC {\"index\":%0d,\"start_cycle\":%0d,\"done_cycle\":%0d,\"latency_cycles\":%0d,\"passive\":true,\"diagnostic_errors\":%0d,\"checks_passed\":",call_index,call_start,cycle_count,cycle_count-call_start,diagnostic_errors);
                if (diagnostic_errors==0) $write("true"); else $write("false");
                $write(",\"engine_fsm_state_cycles\":[");
                for (integer j=0;j<10;j=j+1) begin
                    if (j>0) $write(",");
                    $write("%0d",fsm_cycles[j]);
                end
                $write("],\"stages\":{");
                for (integer j=0;j<5;j=j+1) begin
                    if (j>0) $write(",");
                    $write("\"%0s\":{\"calls\":%0d,\"finishes\":%0d,\"duration_cycles\":{\"sum\":%0d,\"min\":%0d,\"max\":%0d},\"blocked_cycles\":%0d}",stage_name(j),calls[j],finishes[j],sums[j],minima[j],maxima[j],stalls[j]);
                end
                $display("},\"stall_causes_may_overlap\":{\"weight_ar\":%0d,\"weight_r\":%0d,\"store_aw\":%0d,\"store_w\":%0d,\"store_b\":%0d}}",weight_ar,weight_r,store_aw,store_w,store_b);
                active=0; call_index=call_index+1;
            end
            previous_start=stage_start;
        end
    end
endmodule
`undef DE
`undef DI
`undef DA
`undef DW
`undef DC
`undef DS
