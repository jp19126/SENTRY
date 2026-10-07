`timescale 1ns/1ps
// Integration/control test only: real generated control registers, explicit
// arithmetic completion stubs, real mover and a bounded synthetic memory.
module gate_detector_control_tb;
 localparam [63:0] BASE=64'h00000001fffff000;
 localparam integer DEPTH=67108864;
 logic  clk;
 logic  resetn;
 logic  config_valid;
 logic  config_ready;
 logic [63:0] config_arena_base,config_model_id,result_model_id;
 logic [15:0] config_w8_mask;
 logic [31:0] config_threshold_bits;
 logic  window_valid;
 logic  window_ready;
 logic [63:0] window_sample_id;
 logic [63:0] window_document_id;
 logic [63:0] window_arena_base;
 logic [31:0] window_index;
 logic  window_last;
 logic [8:0] window_valid_length;
 logic  result_valid;
 logic  result_ready;
 logic [7:0] result_status;
 logic [63:0] result_sample_id;
 logic [63:0] result_document_id;
 logic [31:0] result_window_index;
 logic [31:0] result_risk_bits;
 logic [31:0] result_max_bits;
 logic  result_last;
 logic  result_reject;
 logic  ctl_target;
 logic [7:0] ctl_awaddr;
 logic [7:0] ctl_araddr;
 logic [31:0] ctl_wdata;
 logic [3:0] ctl_wstrb;
 logic  ctl_awvalid;
 logic  ctl_wvalid;
 logic  ctl_bready;
 logic  ctl_arvalid;
 logic  ctl_rready;
 logic  ctl_awready;
 logic  ctl_wready;
 logic  ctl_bvalid;
 logic  ctl_arready;
 logic  ctl_rvalid;
 logic [1:0] ctl_bresp;
 logic [1:0] ctl_rresp;
 logic [31:0] ctl_rdata;
 logic [1:0] engine_interrupt;
 logic  mover_cmd_valid;
 logic  mover_cmd_ready;
 logic [2:0] mover_cmd_op;
 logic [63:0] mover_cmd_src;
 logic [63:0] mover_cmd_dst;
 logic [15:0] mover_cmd_rows;
 logic [15:0] mover_cmd_cols;
 logic [31:0] mover_cmd_src_stride;
 logic [31:0] mover_cmd_dst_stride;
 logic [31:0] mover_cmd_limit;
 logic  mover_rsp_valid;
 logic  mover_rsp_ready;
 logic [3:0] mover_rsp_status;
 logic [7:0] mover_index_addr;
 logic [7:0] mover_mask_addr;
 logic [31:0] mover_index_value;
 logic [7:0] mover_mask_value;
 logic  mover_rd_valid;
 logic  mover_rd_ready;
 logic [31:0] mover_rd_data;
 logic  mover_rd_last;
 logic [63:0] total_cycles;
 logic [63:0] control_cycles;
 logic [63:0] interrupt_wait_cycles;
 logic [63:0] movement_cycles;
 logic [31:0] control_write_count;
 logic [31:0] control_read_count;
 logic [31:0] integer_calls;
 logic [31:0] fp32_calls;
 logic [15:0] command_index;
 logic [63:0] m_axi_araddr;
 logic [7:0] m_axi_arlen;
 logic [2:0] m_axi_arsize;
 logic [1:0] m_axi_arburst;
 logic m_axi_arvalid;
 logic m_axi_arready;
 logic [31:0] m_axi_rdata;
 logic [1:0] m_axi_rresp;
 logic m_axi_rlast;
 logic m_axi_rvalid;
 logic m_axi_rready;
 logic [63:0] m_axi_awaddr;
 logic [7:0] m_axi_awlen;
 logic [2:0] m_axi_awsize;
 logic [1:0] m_axi_awburst;
 logic m_axi_awvalid;
 logic m_axi_awready;
 logic [31:0] m_axi_wdata;
 logic [3:0] m_axi_wstrb;
 logic m_axi_wlast;
 logic m_axi_wvalid;
 logic m_axi_wready;
 logic [1:0] m_axi_bresp;
 logic m_axi_bvalid;
 logic m_axi_bready;
 logic [2:0] fault_mode;
 logic [1:0] control_fault;
 longint unsigned cycle;
 longint unsigned accepted_cycle,result_elapsed,stub_sum,stub_start[0:1],read_start,write_start;
 logic window_active;
 logic [31:0] desired_risk;
 integer observed_calls,completed_cases;
 logic [31:0] expected[0:8699]; // 435 calls, each 20 exact argument words.
 logic [63:0] stub_pointer[0:1][0:5];
 logic [31:0] stub_rows[0:1],stub_inner[0:1],stub_outputs[0:1],stub_extra[0:1];
 logic [31:0] fp_op,fp_scale;
 logic fp_bias;
 logic [1:0] starts,dones,idles;
 integer stub_ticks[0:1];
 logic [1:0] slave_awready,slave_wready,slave_bvalid,slave_arready,slave_rvalid;
 logic [1:0] slave_bresp[0:1],slave_rresp[0:1];
 logic [31:0] slave_rdata[0:1];
 wire aw_gate=1,w_gate=1,ar_gate=1,b_gate=1,r_gate=1;
 assign ctl_awready=slave_awready[ctl_target]&&aw_gate;
 assign ctl_wready=slave_wready[ctl_target]&&w_gate;
 assign ctl_bvalid=slave_bvalid[ctl_target]&&b_gate;
 assign ctl_bresp=control_fault==1 ? 2'b10:slave_bresp[ctl_target];
 assign ctl_arready=slave_arready[ctl_target]&&ar_gate;
 assign ctl_rvalid=slave_rvalid[ctl_target]&&r_gate;
 assign ctl_rresp=control_fault==2 ? 2'b10:slave_rresp[ctl_target];
 assign ctl_rdata=slave_rdata[ctl_target];
 gate_detector_sequencer dut(.*);
 gate_data_mover mover(.clk(clk),.resetn(resetn),.cmd_valid(mover_cmd_valid),.cmd_ready(mover_cmd_ready),.cmd_op(mover_cmd_op),.cmd_src(mover_cmd_src),.cmd_dst(mover_cmd_dst),.cmd_rows(mover_cmd_rows),.cmd_cols(mover_cmd_cols),.cmd_src_stride(mover_cmd_src_stride),.cmd_dst_stride(mover_cmd_dst_stride),.cmd_limit(mover_cmd_limit),.rsp_valid(mover_rsp_valid),.rsp_ready(mover_rsp_ready),.rsp_status(mover_rsp_status),.index_addr(mover_index_addr),.index_value(mover_index_value),.mask_addr(mover_mask_addr),.mask_value(mover_mask_value),.rd_valid(mover_rd_valid),.rd_ready(mover_rd_ready),.rd_data(mover_rd_data),.rd_last(mover_rd_last),.m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),.m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),.m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),.m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),.m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready),.m_axi_awaddr(m_axi_awaddr),.m_axi_awlen(m_axi_awlen),.m_axi_awsize(m_axi_awsize),.m_axi_awburst(m_axi_awburst),.m_axi_awvalid(m_axi_awvalid),.m_axi_awready(m_axi_awready),.m_axi_wdata(m_axi_wdata),.m_axi_wstrb(m_axi_wstrb),.m_axi_wlast(m_axi_wlast),.m_axi_wvalid(m_axi_wvalid),.m_axi_wready(m_axi_wready),.m_axi_bresp(m_axi_bresp),.m_axi_bvalid(m_axi_bvalid),.m_axi_bready(m_axi_bready));
 gate_mover_axi_memory #(.DEPTH(DEPTH),.BASE(BASE),.STALL(0),.PIPELINED_READ(1)) memory(.*);
 gate_linear_top_control_s_axi registers_0(
  .ACLK(clk),
  .ARESET(!resetn),
  .ACLK_EN(1'b1),
  .AWADDR(ctl_awaddr[6:0]),
  .AWVALID(ctl_awvalid && ctl_target==0 && aw_gate),
  .AWREADY(slave_awready[0]),
  .WDATA(ctl_wdata),
  .WSTRB(ctl_wstrb),
  .WVALID(ctl_wvalid && ctl_target==0 && w_gate),
  .WREADY(slave_wready[0]),
  .BRESP(slave_bresp[0]),
  .BVALID(slave_bvalid[0]),
  .BREADY(ctl_bready && ctl_target==0 && b_gate),
  .ARADDR(ctl_araddr[6:0]),
  .ARVALID(ctl_arvalid && ctl_target==0 && ar_gate),
  .ARREADY(slave_arready[0]),
  .RDATA(slave_rdata[0]),
  .RRESP(slave_rresp[0]),
  .RVALID(slave_rvalid[0]),
  .RREADY(ctl_rready && ctl_target==0 && r_gate),
  .interrupt(engine_interrupt[0]),
  .activations(stub_pointer[0][0]),
  .weights(stub_pointer[0][1]),
  .output_r(stub_pointer[0][2]),
  .rows(stub_rows[0]),
  .inner(stub_inner[0]),
  .outputs(stub_outputs[0]),
  .weight_bits(stub_extra[0]),
  .status(32'd0),
  .status_ap_vld(dones[0]),
  .ap_start(starts[0]),
  .ap_done(dones[0]),
  .ap_ready(dones[0]),
  .ap_idle(idles[0]));
 gate_fixed_fp32_top_control_s_axi registers_1(
  .ACLK(clk),
  .ARESET(!resetn),
  .ACLK_EN(1'b1),
  .AWADDR(ctl_awaddr[7:0]),
  .AWVALID(ctl_awvalid && ctl_target==1 && aw_gate),
  .AWREADY(slave_awready[1]),
  .WDATA(ctl_wdata),
  .WSTRB(ctl_wstrb),
  .WVALID(ctl_wvalid && ctl_target==1 && w_gate),
  .WREADY(slave_wready[1]),
  .BRESP(slave_bresp[1]),
  .BVALID(slave_bvalid[1]),
  .BREADY(ctl_bready && ctl_target==1 && b_gate),
  .ARADDR(ctl_araddr[7:0]),
  .ARVALID(ctl_arvalid && ctl_target==1 && ar_gate),
  .ARREADY(slave_arready[1]),
  .RDATA(slave_rdata[1]),
  .RRESP(slave_rresp[1]),
  .RVALID(slave_rvalid[1]),
  .RREADY(ctl_rready && ctl_target==1 && r_gate),
  .interrupt(engine_interrupt[1]),
  .x(stub_pointer[1][0]),
  .y(stub_pointer[1][1]),
  .z(stub_pointer[1][2]),
  .accumulators(stub_pointer[1][3]),
  .output_r(stub_pointer[1][4]),
  .codes(stub_pointer[1][5]),
  .rows(stub_rows[1]),
  .width(stub_inner[1]),
  .outputs(stub_outputs[1]),
  .op(fp_op),
  .scale(fp_scale),
  .dot_bias(fp_bias),
  .status(32'd0),
  .status_ap_vld(dones[1]),
  .ap_start(starts[1]),
  .ap_done(dones[1]),
  .ap_ready(dones[1]),
  .ap_idle(idles[1]));

 task put_word(input integer offset,input logic [31:0] value);
  begin for(integer b=0;b<4;b=b+1)memory.mem[offset+b]=value[b*8+:8];end
 endtask
 function automatic [31:0] get_word(input integer offset);
  get_word={memory.mem[offset+3],memory.mem[offset+2],memory.mem[offset+1],memory.mem[offset]};
 endfunction
 `include "integration_fixture_constants.svh"
 task initialize_inputs;
  begin
   for(integer i=0;i<256;i=i+1)begin
    put_word(TOKEN_IDS_OFFSET+i*4,i*113%30522);put_word(TYPE_IDS_OFFSET+i*4,i%2);
    memory.mem[ATTENTION_MASK_OFFSET+i]=(i<239);
   end
   for(integer i=0;i<24;i=i+1)put_word(INPUT_SCALES_OFFSET+i*4,32'h3e800000+i);
  end
 endtask
 task check_call(input integer e);
  integer k,b;
  begin
   if(observed_calls>=435)$fatal(1,"Too many service calls");
   b=observed_calls*20;
   if(expected[b]!==e)$fatal(1,"Service order mismatch at %d",observed_calls);
   if(e==0)begin
    for(k=0;k<3;k=k+1)if(stub_pointer[e][k]!=={expected[b+2+k*2],expected[b+1+k*2]})$fatal(1,"Integer pointer mismatch call=%0d pointer=%0d",observed_calls,k);
    if({stub_extra[0],stub_outputs[0],stub_inner[0],stub_rows[0]}!=={expected[b+10],expected[b+9],expected[b+8],expected[b+7]})$fatal(1,"Integer scalar mismatch");
   end else begin
    if(fp_op!==expected[b+1])$fatal(1,"FP32 op mismatch");
    for(k=0;k<6;k=k+1)if(stub_pointer[e][k]!=={expected[b+3+k*2],expected[b+2+k*2]})$fatal(1,"FP32 pointer mismatch call=%0d pointer=%0d",observed_calls,k);
    if({fp_scale,stub_outputs[1],stub_inner[1],stub_rows[1]}!=={expected[b+17],expected[b+16],expected[b+15],expected[b+14]} || fp_bias!==expected[b+18][0])$fatal(1,"FP32 scalar mismatch at %0d",observed_calls);
   end
   observed_calls=observed_calls+1;
  end
 endtask
 always #2.5 clk=~clk;
 always @(posedge clk)begin
  if(!resetn)begin cycle<=0;window_active=0;stub_sum=0;result_elapsed=0;dones<=0;idles<=3;stub_ticks[0]<=0;stub_ticks[1]<=0;end
  else begin
   cycle<=cycle+1;dones<=0;
   if(window_valid&&window_ready)begin
    accepted_cycle=cycle;window_active=1;stub_sum=0;result_elapsed=0;read_start=memory.read_beats;write_start=memory.write_beats;
   end
   if(result_valid&&window_active)begin result_elapsed=cycle-accepted_cycle;window_active=0;end
   if(cycle>120000000)$fatal(1,"Bounded integration cycle timeout");
   for(integer e=0;e<2;e=e+1)begin
    if(dones[e])begin
     if(dut.state!=12)$fatal(1,"Stub completed before IRQ_WAIT; substitution boundary invalid");
     stub_sum=stub_sum+cycle-stub_start[e];
     $display("GATE_INTEGRATION_STUB {\"case_index\":%0d,\"call_index\":%0d,\"engine\":%0d,\"start_cycle\":%0d,\"done_cycle\":%0d,\"latency_cycles\":%0d}",completed_cases,observed_calls-1,e,stub_start[e],cycle,cycle-stub_start[e]);
    end
    if(starts[e]&&idles[e]&&!dones[e])begin
     check_call(e);stub_start[e]=cycle;idles[e]<=0;stub_ticks[e]<=7;
    end else if(stub_ticks[e]>1)stub_ticks[e]<=stub_ticks[e]-1;
    else if(stub_ticks[e]==1)begin
     stub_ticks[e]<=0;dones[e]<=1;idles[e]<=1;
     // Only the final probability is injected. No detector math is claimed.
     if(e==1 && fp_op==6 && stub_rows[e]==1 && stub_inner[e]==2)
      put_word(integer'(stub_pointer[e][4]-BASE)+4,desired_risk);
    end
   end
  end
 end
 task reset_and_configure;
  begin
   @(negedge clk);resetn=0;config_valid=0;window_valid=0;result_ready=0;
   fault_mode=0;control_fault=0;observed_calls=0;
   repeat(5)@(negedge clk);resetn=1;
   config_w8_mask=16'ha55a;config_threshold_bits=32'h3b955a95;config_arena_base=BASE;config_model_id=64'hbf04010000000001;config_valid=1;
   @(posedge clk);if(!config_ready)$fatal(1,"Configuration not accepted");@(negedge clk);config_valid=0;
   wait(window_ready);@(negedge clk);
  end
 endtask
 task submit(input logic[63:0] doc,input logic[31:0] idx,input logic last,input logic[63:0] base,input logic[8:0] n);
  begin
   wait(window_ready);@(negedge clk);window_document_id=doc;window_sample_id=64'hfedcba9876540000+completed_cases;
   window_index=idx;window_last=last;window_arena_base=base;window_valid_length=n;window_valid=1;
   @(posedge clk);if(!window_ready)$fatal(1,"Window not accepted");@(negedge clk);window_valid=0;
  end
 endtask
 task consume(input string name,input integer status,input logic[31:0] risk,input logic[31:0] maximum,input logic reject_bit,input bit complete_window);
  longint unsigned rb,wb;
  logic[31:0] saved_max,saved_risk;
  begin
   wait(result_valid);@(posedge clk);@(negedge clk);
   if(result_status!==status || result_sample_id!==window_sample_id || result_document_id!==window_document_id || result_window_index!==window_index || result_last!==window_last || result_model_id!==config_model_id)$fatal(1,"Result identity/status mismatch %s status=%0d",name,result_status);
   if(result_risk_bits!==risk || result_max_bits!==maximum || result_reject!==reject_bit)$fatal(1,"Result probability mismatch %s risk=%h max=%h reject=%b",name,result_risk_bits,result_max_bits,result_reject);
   if(complete_window && (integer_calls!=24 || fp32_calls!=411 || observed_calls!=435 || control_write_count!=8508 || control_read_count!=870))$fatal(1,"Full control counts mismatch %s int=%0d fp=%0d calls=%0d wr=%0d rd=%0d",name,integer_calls,fp32_calls,observed_calls,control_write_count,control_read_count);
   if(result_elapsed!=total_cycles+1)$fatal(1,"Inclusive timing boundary differs");
   if(complete_window && (memory.read_beats-read_start!=2228825 || memory.write_beats-write_start!=2260992))$fatal(1,"Planned mover traffic mismatch reads=%0d writes=%0d",memory.read_beats-read_start,memory.write_beats-write_start);
   saved_max=result_max_bits;saved_risk=result_risk_bits;
   repeat(3)begin @(negedge clk);if(!result_valid || result_max_bits!==saved_max || result_risk_bits!==saved_risk)$fatal(1,"Result changed under backpressure");end
   $display("GATE_INTEGRATION_CASE {\"id\":\"%s\",\"status\":%0d,\"complete_window\":%s,\"total_cycles\":%0d,\"inclusive_cycles\":%0d,\"stub_interval_cycles\":%0d,\"control_layout_cycles\":%0d,\"control_cycles\":%0d,\"interrupt_wait_cycles\":%0d,\"movement_cycles\":%0d,\"control_writes\":%0d,\"control_reads\":%0d,\"integer_calls\":%0d,\"fp32_calls\":%0d,\"read_beats\":%0d,\"write_beats\":%0d}",name,status,complete_window?"true":"false",total_cycles,result_elapsed,stub_sum,result_elapsed-stub_sum,control_cycles,interrupt_wait_cycles,movement_cycles,control_write_count,control_read_count,integer_calls,fp32_calls,memory.read_beats-read_start,memory.write_beats-write_start);
   completed_cases=completed_cases+1;result_ready=1;@(negedge clk);result_ready=0;
   if(status!=0)begin repeat(3)@(negedge clk);if(window_ready || mover_cmd_valid || ctl_awvalid || ctl_arvalid)$fatal(1,"Error did not halt until reset");end
  end
 endtask
 initial begin
  clk=0;resetn=0;config_valid=0;window_valid=0;result_ready=0;completed_cases=0;observed_calls=0;fault_mode=0;control_fault=0;
  $readmemh("expected_arguments.mem",expected);
  // Nonzero deterministic bit patterns make layout traffic observable, without
  // treating these words as arithmetic inputs to the completion stubs.
  for(integer i=0;i<DEPTH;i=i+4)put_word(i,32'h3f000000+(i>>2)%65536);
  initialize_inputs();reset_and_configure();
  desired_risk=32'h3b955a96;submit(77,0,0,BASE,239);consume("document_first_above",0,desired_risk,desired_risk,0,1);
  observed_calls=0;desired_risk=32'h3b955a95;submit(77,1,1,BASE,239);consume("document_last_preserves_max",0,desired_risk,32'h3b955a96,1,1);
  observed_calls=0;desired_risk=32'h3b955a95;submit(88,0,1,BASE,239);consume("new_document_equal_threshold",0,desired_risk,desired_risk,0,1);
  submit(88,2,1,BASE,239);consume("noncontiguous_index",2,0,0,0,0);
  reset_and_configure();submit(99,0,1,64'hfffffffffffff000,239);consume("arena_overflow",2,0,0,0,0);
  reset_and_configure();submit(99,0,1,BASE+1,239);consume("arena_alignment",2,0,0,0,0);
  reset_and_configure();submit(99,0,1,BASE,0);consume("invalid_valid_length",2,0,0,0,0);
  reset_and_configure();put_word(TOKEN_IDS_OFFSET,32'h80000001);submit(99,0,1,BASE,239);consume("full_width_token_id",3,0,0,0,0);initialize_inputs();
  reset_and_configure();put_word(TYPE_IDS_OFFSET,32'h80000000);submit(99,0,1,BASE,239);consume("full_width_type_id",3,0,0,0,0);initialize_inputs();
  reset_and_configure();memory.mem[ATTENTION_MASK_OFFSET]=2;submit(99,0,1,BASE,239);consume("binary_mask_validation",3,0,0,0,0);initialize_inputs();
  reset_and_configure();put_word(INPUT_SCALES_OFFSET,32'h7fc00000);submit(99,0,1,BASE,239);consume("finite_scale_validation",3,0,0,0,0);initialize_inputs();
  reset_and_configure();fault_mode=1;submit(99,0,1,BASE,239);consume("mover_read_error",4,0,0,0,0);
  reset_and_configure();desired_risk=32'h7fc00000;submit(99,0,1,BASE,239);consume("nonfinite_final_risk",7,0,0,0,1);
  if(completed_cases!=13)$fatal(1,"Missing integration case");
  $display("PASS gate_detector_control 13 cases; arithmetic completion stubs only");$finish;
 end
endmodule
