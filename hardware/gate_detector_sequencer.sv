`timescale 1ns/1ps
// Fixed BERT-Mini L256 control, not a general command processor.
// The ROM comes only from scripts/plan_detector_commands.py. Arithmetic engines
// remain the existing HLS modules. No arithmetic or physical-DDR timing is here.
module gate_detector_sequencer (
 input wire clk, resetn,
 input wire config_valid, output wire config_ready,
 input wire [15:0] config_w8_mask, input wire [31:0] config_threshold_bits,
 input wire [63:0] config_arena_base, config_model_id,
 input wire window_valid, output wire window_ready,
 input wire [63:0] window_sample_id, window_document_id, window_arena_base,
 input wire [31:0] window_index, input wire window_last,
 input wire [8:0] window_valid_length,
 output wire result_valid, input wire result_ready,
 output reg [7:0] result_status,
 output reg [63:0] result_sample_id, result_document_id, result_model_id,
 output reg [31:0] result_window_index, result_risk_bits, result_max_bits,
 output reg result_last, result_reject,
 // Single AXI-Lite master; target selects the integer (0) or FP32 (1) port.
 output wire ctl_target, output wire [7:0] ctl_awaddr, ctl_araddr,
 output wire [31:0] ctl_wdata, output wire [3:0] ctl_wstrb,
 output wire ctl_awvalid, ctl_wvalid, ctl_bready, ctl_arvalid, ctl_rready,
 input wire ctl_awready, ctl_wready, ctl_bvalid, ctl_arready, ctl_rvalid,
 input wire [1:0] ctl_bresp, ctl_rresp, input wire [31:0] ctl_rdata,
 input wire [1:0] engine_interrupt,
 output wire mover_cmd_valid, input wire mover_cmd_ready,
 output reg [2:0] mover_cmd_op,
 output reg [63:0] mover_cmd_src, mover_cmd_dst,
 output reg [15:0] mover_cmd_rows, mover_cmd_cols,
 output reg [31:0] mover_cmd_src_stride, mover_cmd_dst_stride, mover_cmd_limit,
 input wire mover_rsp_valid, output wire mover_rsp_ready, input wire [3:0] mover_rsp_status,
 input wire [7:0] mover_index_addr, mover_mask_addr,
 output wire [31:0] mover_index_value, output wire [7:0] mover_mask_value,
 input wire mover_rd_valid, output wire mover_rd_ready,
 input wire [31:0] mover_rd_data, input wire mover_rd_last,
 output reg [63:0] total_cycles, control_cycles, interrupt_wait_cycles, movement_cycles,
 output reg [31:0] control_write_count, control_read_count, integer_calls, fp32_calls,
 output reg [15:0] command_index
);
 `include "command_plan_constants.svh"
 reg [511:0] plan_rom [0:PLAN_COUNT-1];
 initial $readmemh("command_plan.mem", plan_rom);
 reg [511:0] descriptor;
 function [31:0] word_at(input integer n); word_at=descriptor[n*32 +:32]; endfunction
 function valid_probability(input [31:0] bits);
  valid_probability=(!bits[31] && bits<=32'h3f800000) || bits==32'h80000000;
 endfunction
 function [31:0] canonical_zero(input [31:0] bits);
  canonical_zero=(bits[30:0]==0) ? 32'd0 : bits;
 endfunction
 localparam CFG=0, SETUP_SEND=1, SETUP_WAIT=2, IDLE=3, LOAD_PREP=4,
  MOVE_SEND=5, MOVE_WAIT=6, LOAD_CHECK=7, FETCH=8, DISPATCH=9,
  ARG_SEND=10, ARG_WAIT=11, IRQ_WAIT=12, DONE_SEND=13, DONE_WAIT=14,
  STATUS_SEND=15, STATUS_WAIT=16, CLEAR_SEND=17, CLEAR_WAIT=18, RESULT=19, HALTED=20;
 reg [4:0] state;
 reg [15:0] frozen_w8;
 reg [31:0] threshold, document_max, expected_window;
 reg [63:0] arena_base, deployed_base, active_document, active_base;
 reg document_active, selected_engine, index_is_type;
 reg [8:0] valid_length;
 reg [1:0] setup_index, load_index;
 reg [1:0] move_role; // 0 input read, 1 layout operation, 2 final risk read.
 reg [8:0] received_words;
 reg [9:0] mask_sum;
 reg input_bad, stream_bad;
 reg [31:0] risk_received;
 reg [4:0] argument_index;
 reg [31:0] token_cache[0:255], type_cache[0:255], scale_cache[0:23];
 reg [7:0] mask_cache[0:255];
 assign mover_index_value=index_is_type ? type_cache[mover_index_addr] : token_cache[mover_index_addr];
 assign mover_mask_value=mask_cache[mover_mask_addr];
 assign config_ready=(state==CFG);
 assign window_ready=(state==IDLE);
 assign result_valid=(state==RESULT);
 assign mover_cmd_valid=(state==MOVE_SEND);
 assign mover_rsp_ready=(state==MOVE_WAIT);
 assign mover_rd_ready=(state==MOVE_WAIT && move_role!=1);

 reg bus_req_valid, bus_req_write, bus_req_target;
 reg [7:0] bus_req_address;
 reg [31:0] bus_req_data;
 wire bus_req_ready, bus_rsp_valid;
 wire [1:0] bus_rsp_error;
 wire [31:0] bus_rsp_data;
 wire bus_rsp_ready=(state==SETUP_WAIT || state==ARG_WAIT || state==DONE_WAIT || state==STATUS_WAIT || state==CLEAR_WAIT);
 gate_control_master control_master (
 .clk(clk),.resetn(resetn),.req_valid(bus_req_valid),.req_ready(bus_req_ready),
 .req_write(bus_req_write),.req_target(bus_req_target),.req_address(bus_req_address),.req_data(bus_req_data),
 .rsp_valid(bus_rsp_valid),.rsp_ready(bus_rsp_ready),.rsp_error(bus_rsp_error),.rsp_data(bus_rsp_data),
 .target(ctl_target),.awaddr(ctl_awaddr),.awvalid(ctl_awvalid),.awready(ctl_awready),
 .wdata(ctl_wdata),.wstrb(ctl_wstrb),.wvalid(ctl_wvalid),.wready(ctl_wready),
 .bresp(ctl_bresp),.bvalid(ctl_bvalid),.bready(ctl_bready),
 .araddr(ctl_araddr),.arvalid(ctl_arvalid),.arready(ctl_arready),
 .rdata(ctl_rdata),.rresp(ctl_rresp),.rvalid(ctl_rvalid),.rready(ctl_rready));

 reg [31:0] relative_pointer;
 reg [63:0] absolute_pointer;
 reg pointer_high;
 integer pointer_number;
 always @* begin
  bus_req_valid=0; bus_req_write=1; bus_req_target=selected_engine;
  bus_req_address=0; bus_req_data=0; relative_pointer=0; absolute_pointer=0;
  pointer_high=0; pointer_number=0;
  case (state)
   SETUP_SEND: begin
    bus_req_valid=1; bus_req_target=setup_index[1];
    bus_req_address=setup_index[0] ? 8'h08 : 8'h04; bus_req_data=1;
   end
   ARG_SEND: begin
    bus_req_valid=1;
    if (!selected_engine) begin
     if (argument_index<6) begin
      pointer_number=argument_index>>1; pointer_high=argument_index[0];
      relative_pointer=word_at(1+pointer_number);
      case (pointer_number)
       0:bus_req_address=8'h10+(pointer_high?4:0);
       1:bus_req_address=8'h1c+(pointer_high?4:0);
       2:bus_req_address=8'h28+(pointer_high?4:0);
      endcase
     end else case (argument_index)
      6:begin bus_req_address=8'h34; bus_req_data=word_at(4); end
      7:begin bus_req_address=8'h3c; bus_req_data=word_at(5); end
      8:begin bus_req_address=8'h44; bus_req_data=word_at(6); end
      9:begin bus_req_address=8'h4c; bus_req_data=frozen_w8[word_at(7)]?8:4; end
      10:begin bus_req_address=0; bus_req_data=1; end
     endcase
    end else begin
     if (argument_index==0) begin bus_req_address=8'h10;bus_req_data=word_at(1); end
     else if (argument_index<=12) begin
      pointer_number=(argument_index-1)>>1; pointer_high=!(argument_index[0]);
      relative_pointer=word_at(2+pointer_number);
      if (pointer_number==1 && word_at(14)<16 && frozen_w8[word_at(14)]) relative_pointer=word_at(15);
      case (pointer_number)
       0:bus_req_address=8'h18+(pointer_high?4:0);
       1:bus_req_address=8'h24+(pointer_high?4:0);
       2:bus_req_address=8'h30+(pointer_high?4:0);
       3:bus_req_address=8'h3c+(pointer_high?4:0);
       4:bus_req_address=8'h48+(pointer_high?4:0);
       5:bus_req_address=8'h54+(pointer_high?4:0);
      endcase
     end else case (argument_index)
      13:begin bus_req_address=8'h60;bus_req_data=word_at(8);end
      14:begin bus_req_address=8'h68;bus_req_data=word_at(9);end
      15:begin bus_req_address=8'h70;bus_req_data=word_at(10);end
      16:begin bus_req_address=8'h78;bus_req_data=word_at(12)<24 ? scale_cache[word_at(12)] : word_at(11);end
      17:begin bus_req_address=8'h80;bus_req_data=word_at(13);end
      18:begin bus_req_address=0;bus_req_data=1;end
     endcase
    end
    if ((!selected_engine && argument_index<6) || (selected_engine && argument_index>=1 && argument_index<=12)) begin
     // Base was checked with the entire 64 MiB extent before any dispatch.
     // The addition must be 64 bit before extracting the low/high registers.
     absolute_pointer=arena_base+{32'd0,relative_pointer};
     bus_req_data=pointer_high ? absolute_pointer[63:32] : absolute_pointer[31:0];
    end
   end
   DONE_SEND:begin bus_req_valid=1;bus_req_write=0;bus_req_address=0;end
   STATUS_SEND:begin bus_req_valid=1;bus_req_write=0;bus_req_address=selected_engine?8'h88:8'h54;end
   CLEAR_SEND:begin bus_req_valid=1;bus_req_address=8'h0c;bus_req_data=1;end
   default:begin end
  endcase
 end
 task fail(input [7:0] code);
 begin result_status<=code;result_reject<=0;result_risk_bits<=0;result_max_bits<=0;
       document_active<=0;document_max<=0;expected_window<=0;state<=RESULT;end
 endtask
 integer j;
 reg [64:0] arena_end;
 always @(posedge clk) begin
  if (!resetn) begin
   state<=CFG;setup_index<=0;selected_engine<=0;document_active<=0;document_max<=0;
   expected_window<=0;threshold<=0;frozen_w8<=0;arena_base<=0;active_base<=0;active_document<=0;
   result_status<=0;result_sample_id<=0;result_document_id<=0;result_model_id<=0;deployed_base<=0;result_window_index<=0;
   result_risk_bits<=0;result_max_bits<=0;result_last<=0;result_reject<=0;
   total_cycles<=0;control_cycles<=0;interrupt_wait_cycles<=0;movement_cycles<=0;
   control_write_count<=0;control_read_count<=0;integer_calls<=0;fp32_calls<=0;
   command_index<=0;argument_index<=0;load_index<=0;move_role<=0;received_words<=0;
   mask_sum<=0;input_bad<=0;stream_bad<=0;risk_received<=0;index_is_type<=0;valid_length<=0;
   mover_cmd_op<=0;mover_cmd_src<=0;mover_cmd_dst<=0;mover_cmd_rows<=0;mover_cmd_cols<=0;
   mover_cmd_src_stride<=0;mover_cmd_dst_stride<=0;mover_cmd_limit<=0;
  end else begin
   if (state>=LOAD_PREP && state<RESULT) begin
    total_cycles<=total_cycles+1;
    if (state==ARG_SEND || state==ARG_WAIT || state==DONE_SEND || state==DONE_WAIT || state==STATUS_SEND || state==STATUS_WAIT || state==CLEAR_SEND || state==CLEAR_WAIT)
     control_cycles<=control_cycles+1;
    if (state==IRQ_WAIT) interrupt_wait_cycles<=interrupt_wait_cycles+1;
    if (state==MOVE_SEND || state==MOVE_WAIT) movement_cycles<=movement_cycles+1;
   end
   if (bus_req_valid && bus_req_ready && state!=SETUP_SEND) begin
    if(bus_req_write) control_write_count<=control_write_count+1;else control_read_count<=control_read_count+1;
   end
   if(mover_rd_valid && mover_rd_ready) begin
    received_words<=received_words+1;
    if(mover_rd_last != (received_words+1==mover_cmd_cols)) stream_bad<=1;
    if (received_words>=mover_cmd_cols) stream_bad<=1;
    else if(move_role==2) risk_received<=mover_rd_data;
    else case(load_index)
     0:begin token_cache[received_words[7:0]]<=mover_rd_data;if(mover_rd_data>=30522) input_bad<=1;end
     1:begin type_cache[received_words[7:0]]<=mover_rd_data;if(mover_rd_data>=2) input_bad<=1;end
     2:begin
      for(j=0;j<4;j=j+1) begin
       mask_cache[received_words*4+j]<=mover_rd_data[j*8+:8];
       if(mover_rd_data[j*8+:8]>1) input_bad<=1;
      end
      mask_sum<=mask_sum+{9'd0,mover_rd_data[0]}+{9'd0,mover_rd_data[8]}+{9'd0,mover_rd_data[16]}+{9'd0,mover_rd_data[24]};
     end
     3:begin scale_cache[received_words]<=mover_rd_data;
      if(mover_rd_data[31] || mover_rd_data[30:0]==0 || mover_rd_data[30:23]==8'hff) input_bad<=1;
     end
    endcase
   end
   case(state)
    CFG:if(config_valid) begin
     arena_end={1'b0,config_arena_base}+{1'b0,ARENA_BYTES};
     if(!valid_probability(config_threshold_bits) || config_arena_base[11:0]!=0 || arena_end[64]) begin result_status<=1;state<=RESULT;end
     else begin threshold<=canonical_zero(config_threshold_bits);frozen_w8<=config_w8_mask;deployed_base<=config_arena_base;result_model_id<=config_model_id;setup_index<=0;state<=SETUP_SEND;end
    end
    SETUP_SEND:if(bus_req_ready)state<=SETUP_WAIT;
    SETUP_WAIT:if(bus_rsp_valid)begin
     if(bus_rsp_error!=0)fail(5);
     else if(setup_index==3)state<=IDLE;
     else begin setup_index<=setup_index+1;state<=SETUP_SEND;end
    end
    IDLE:if(window_valid)begin
     result_sample_id<=window_sample_id;result_document_id<=window_document_id;
     result_window_index<=window_index;result_last<=window_last;result_status<=0;result_reject<=0;
     result_risk_bits<=0;result_max_bits<=0;
     total_cycles<=0;control_cycles<=0;interrupt_wait_cycles<=0;movement_cycles<=0;
     control_write_count<=0;control_read_count<=0;integer_calls<=0;fp32_calls<=0;
     arena_end={1'b0,window_arena_base}+{1'b0,ARENA_BYTES};
     if(window_arena_base!=deployed_base || window_arena_base[11:0]!=0 || arena_end[64] || window_valid_length==0 || window_valid_length>256 ||
        (!document_active && window_index!=0) || (document_active && (window_document_id!=active_document || window_arena_base!=active_base || window_index!=expected_window)) ||
        (!window_last && window_index==32'hffffffff)) fail(2);
     else begin
      arena_base<=window_arena_base;valid_length<=window_valid_length;active_document<=window_document_id;active_base<=window_arena_base;
      if(!document_active)document_max<=0;
      document_active<=1;load_index<=0;mask_sum<=0;input_bad<=0;state<=LOAD_PREP;
     end
    end
    LOAD_PREP:begin
     mover_cmd_op<=4;mover_cmd_rows<=1;mover_cmd_dst<=0;mover_cmd_src_stride<=0;mover_cmd_dst_stride<=0;mover_cmd_limit<=0;
     case(load_index)
      0:begin mover_cmd_src<=arena_base+TOKEN_IDS_OFFSET;mover_cmd_cols<=256;end
      1:begin mover_cmd_src<=arena_base+TYPE_IDS_OFFSET;mover_cmd_cols<=256;end
      2:begin mover_cmd_src<=arena_base+ATTENTION_MASK_OFFSET;mover_cmd_cols<=64;end
      3:begin mover_cmd_src<=arena_base+INPUT_SCALES_OFFSET;mover_cmd_cols<=24;end
     endcase
     received_words<=0;stream_bad<=0;move_role<=0;state<=MOVE_SEND;
    end
    MOVE_SEND:if(mover_cmd_ready)state<=MOVE_WAIT;
    MOVE_WAIT:if(mover_rsp_valid)begin
     if(mover_rsp_status!=0)fail(4);
     else if(move_role!=1 && (stream_bad || received_words!=mover_cmd_cols))fail(8);
     else if(move_role==0)begin
      if(load_index==3)state<=LOAD_CHECK;
      else begin load_index<=load_index+1;state<=LOAD_PREP;end
     end else if(move_role==1)begin command_index<=command_index+1;state<=FETCH;end
     else if(!valid_probability(risk_received))fail(7);
     else begin
      result_risk_bits<=canonical_zero(risk_received);
      result_max_bits<=canonical_zero(risk_received)>document_max ? canonical_zero(risk_received) : document_max;
      document_max<=canonical_zero(risk_received)>document_max ? canonical_zero(risk_received) : document_max;
      result_reject<=result_last && ((canonical_zero(risk_received)>document_max ? canonical_zero(risk_received) : document_max)>threshold);
      expected_window<=result_window_index+1;
      if(result_last)begin document_active<=0;document_max<=0;expected_window<=0;end
      state<=RESULT;
     end
    end
    LOAD_CHECK:if(input_bad || mask_sum!={1'b0,valid_length})fail(3);
     else begin command_index<=1;state<=FETCH;end
    FETCH:if(command_index>=PLAN_COUNT)fail(8);
     else begin descriptor<=plan_rom[command_index];state<=DISPATCH;end
    DISPATCH:case(word_at(0))
     1,2:begin selected_engine<=(word_at(0)==2);argument_index<=0;state<=ARG_SEND;end
     3:begin
      mover_cmd_op<=word_at(1);mover_cmd_src<=arena_base+{32'd0,word_at(2)};mover_cmd_dst<=arena_base+{32'd0,word_at(3)};
      mover_cmd_rows<=word_at(4);mover_cmd_cols<=word_at(5);mover_cmd_src_stride<=word_at(6);mover_cmd_dst_stride<=word_at(7);
      mover_cmd_limit<=word_at(8);index_is_type<=word_at(9);move_role<=1;state<=MOVE_SEND;
     end
     4:begin
      mover_cmd_op<=4;mover_cmd_src<=arena_base+{32'd0,word_at(1)};mover_cmd_dst<=0;mover_cmd_rows<=1;mover_cmd_cols<=1;
      mover_cmd_src_stride<=0;mover_cmd_dst_stride<=0;mover_cmd_limit<=0;received_words<=0;stream_bad<=0;move_role<=2;state<=MOVE_SEND;
     end
     default:fail(8);
    endcase
    ARG_SEND:if(bus_req_ready)state<=ARG_WAIT;
    ARG_WAIT:if(bus_rsp_valid)begin
     if(bus_rsp_error!=0)fail(5);
     else if(argument_index==(selected_engine?18:10))begin
      if(selected_engine)fp32_calls<=fp32_calls+1;else integer_calls<=integer_calls+1;
      state<=IRQ_WAIT;
     end else begin argument_index<=argument_index+1;state<=ARG_SEND;end
    end
    IRQ_WAIT:if(engine_interrupt[selected_engine])state<=DONE_SEND;
    DONE_SEND:if(bus_req_ready)state<=DONE_WAIT;
    DONE_WAIT:if(bus_rsp_valid)begin
     if(bus_rsp_error!=0)fail(5);else if(!bus_rsp_data[1])fail(6);else state<=STATUS_SEND;
    end
    STATUS_SEND:if(bus_req_ready)state<=STATUS_WAIT;
    STATUS_WAIT:if(bus_rsp_valid)begin
     if(bus_rsp_error!=0)fail(5);else if(bus_rsp_data!=0)fail(6);else state<=CLEAR_SEND;
    end
    CLEAR_SEND:if(bus_req_ready)state<=CLEAR_WAIT;
    CLEAR_WAIT:if(bus_rsp_valid)begin
     if(bus_rsp_error!=0)fail(5);else begin command_index<=command_index+1;state<=FETCH;end
    end
    RESULT:if(result_ready)begin
     // Every error requires reset, preventing stale IRQ or partial output reuse.
     state<=(result_status!=0) ? HALTED : IDLE;
    end
    HALTED:begin end
    default:fail(8);
   endcase
  end
 end
endmodule

// One request at a time. AW and W independently remain valid until accepted;
// response backpressure cannot change payload or target. No fixed bus latency.
module gate_control_master (
 input wire clk,resetn,req_valid,output wire req_ready,
 input wire req_write,req_target,input wire [7:0] req_address,input wire [31:0] req_data,
 output wire rsp_valid,input wire rsp_ready,output reg [1:0] rsp_error,output reg [31:0] rsp_data,
 output reg target,output reg [7:0] awaddr,araddr,output reg [31:0] wdata,
 output wire [3:0] wstrb,output reg awvalid,wvalid,output wire bready,
 input wire awready,wready,input wire [1:0] bresp,input wire bvalid,
 output reg arvalid,input wire arready,input wire [31:0] rdata,input wire [1:0] rresp,input wire rvalid,output wire rready
);
 localparam IDLE=0,WRITE=1,READ=2,RESPONSE=3;
 reg [1:0] state;
 assign req_ready=(state==IDLE);assign rsp_valid=(state==RESPONSE);assign wstrb=4'hf;
 assign bready=(state==WRITE && !awvalid && !wvalid);
 assign rready=(state==READ && !arvalid);
 always @(posedge clk)begin
  if(!resetn)begin state<=IDLE;target<=0;awaddr<=0;araddr<=0;wdata<=0;awvalid<=0;wvalid<=0;arvalid<=0;rsp_error<=0;rsp_data<=0;end
  else case(state)
   IDLE:if(req_valid)begin
    target<=req_target;awaddr<=req_address;araddr<=req_address;wdata<=req_data;
    if(req_write)begin awvalid<=1;wvalid<=1;state<=WRITE;end
    else begin arvalid<=1;state<=READ;end
   end
   WRITE:begin
    if(awvalid&&awready)awvalid<=0;if(wvalid&&wready)wvalid<=0;
    if(bvalid&&bready)begin rsp_error<=bresp;rsp_data<=0;state<=RESPONSE;end
   end
   READ:begin
    if(arvalid&&arready)arvalid<=0;
    if(rvalid&&rready)begin rsp_error<=rresp;rsp_data<=rdata;state<=RESPONSE;end
   end
   RESPONSE:if(rsp_ready)state<=IDLE;
  endcase
 end
endmodule
