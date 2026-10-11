`timescale 1ns/1ps
`ifndef GATE_TOKENS
`define GATE_TOKENS 256
`endif
module gate_data_mover_tb;
 localparam integer TOKENS=`GATE_TOKENS;
 localparam integer DEPTH=TOKENS==256 ? 262144 : 2097152;
 logic clk, resetn;
 logic cmd_valid; logic cmd_ready;
 logic [2:0] cmd_op;
 logic [63:0] cmd_src, cmd_dst;
 logic [15:0] cmd_rows, cmd_cols;
 logic [31:0] cmd_src_stride, cmd_dst_stride, cmd_limit;
 logic rsp_valid; logic rsp_ready; logic [3:0] rsp_status;
 logic [$clog2(TOKENS)-1:0] index_addr; logic [31:0] index_value;
 logic [$clog2(TOKENS)-1:0] mask_addr; logic [7:0] mask_value;
 logic rd_valid; logic rd_ready;
 logic [31:0] rd_data; logic rd_last;
 logic [63:0] m_axi_araddr; logic [7:0] m_axi_arlen;
 logic [2:0] m_axi_arsize; logic [1:0] m_axi_arburst;
 logic m_axi_arvalid; logic m_axi_arready;
 logic [31:0] m_axi_rdata; logic [1:0] m_axi_rresp;
 logic m_axi_rlast, m_axi_rvalid; logic m_axi_rready;
 logic [63:0] m_axi_awaddr; logic [7:0] m_axi_awlen;
 logic [2:0] m_axi_awsize; logic [1:0] m_axi_awburst;
 logic m_axi_awvalid; logic m_axi_awready;
 logic [31:0] m_axi_wdata; logic [3:0] m_axi_wstrb;
 logic m_axi_wlast, m_axi_wvalid; logic m_axi_wready;
 logic [1:0] m_axi_bresp; logic m_axi_bvalid;
 logic m_axi_bready;

 localparam logic [63:0] BASE=64'h100000000;
 logic [2:0] fault_mode;
 logic [31:0] indices[0:TOKENS-1];
 logic [7:0] masks[0:TOKENS-1];
 integer stream_count,stream_expected,stream_start,checks;
 logic check_stream;
 longint unsigned cycle;
 gate_data_mover #(.TOKENS(TOKENS)) dut(.*);
 gate_mover_axi_memory #(.BASE(BASE),.DEPTH(DEPTH)) memory(.*);
 assign index_value=indices[index_addr];
 assign mask_value=masks[mask_addr];
 assign rd_ready=cycle%3!=0;
 initial begin clk=0;forever #2.5 clk=~clk;end
 always @(posedge clk)begin
  if(!resetn)cycle<=0;else cycle<=cycle+1;
  if(resetn&&check_stream&&rd_valid&&rd_ready)begin
   if(rd_data!==word_at(stream_start+4*stream_count))$fatal(1,"Stream data mismatch");
   if(rd_last!==(stream_count==stream_expected-1))$fatal(1,"Stream last mismatch");
   stream_count=stream_count+1;
  end
 end
 function automatic logic [31:0] word_at(input integer a);
  word_at={memory.mem[a+3],memory.mem[a+2],memory.mem[a+1],memory.mem[a]};
 endfunction
 task automatic command(input string name,input integer operation,input integer so,doff,
    r,c,sstride,dstride,lim,expected_status,
    input logic [63:0] source_base=BASE,destination_base=BASE);
  longint unsigned start,stop,rb,wb,rr,ww;
  logic [3:0] held_status;
  begin
   @(negedge clk);while(!cmd_ready)@(negedge clk);
   cmd_op=operation;cmd_src=source_base+so;cmd_dst=destination_base+doff;cmd_rows=r;cmd_cols=c;
   cmd_src_stride=sstride;cmd_dst_stride=dstride;cmd_limit=lim;
   rb=memory.read_beats;wb=memory.write_beats;rr=memory.read_bursts;ww=memory.write_bursts;
   cmd_valid=1;@(posedge clk);start=cycle+1; // counter value after the accepted-command edge
@(negedge clk);cmd_valid=0;
   while(!rsp_valid)@(negedge clk);stop=cycle;
   if(rsp_status!==expected_status[3:0])$fatal(1,"%s expected status %0d got %0d",name,expected_status,rsp_status);
   if(memory.read_active||memory.write_active||m_axi_rvalid||m_axi_bvalid)$fatal(1,"Completion left a transaction outstanding");
   held_status=rsp_status;repeat(3)begin @(negedge clk);if(!rsp_valid||rsp_status!==held_status||cmd_ready)$fatal(1,"Completion changed under backpressure");end
   $display("GATE_MOVER_CASE {\"id\":\"%s\",\"cycles\":%0d,\"status\":%0d,\"read_bursts\":%0d,\"read_beats\":%0d,\"write_bursts\":%0d,\"write_beats\":%0d}",name,stop-start,rsp_status,memory.read_bursts-rr,memory.read_beats-rb,memory.write_bursts-ww,memory.write_beats-wb);
   rsp_ready=1;@(negedge clk);rsp_ready=0;checks=checks+1;
  end
 endtask
 initial begin : test
  integer i,j,r,c;
  logic [31:0] actual_word,expected_word;
  resetn=0;cmd_valid=0;rsp_ready=0;cmd_op=0;cmd_src=0;cmd_dst=0;
  cmd_rows=0;cmd_cols=0;cmd_src_stride=0;cmd_dst_stride=0;cmd_limit=0;
  fault_mode=0;check_stream=0;stream_count=0;checks=0;
  for(i=0;i<DEPTH;i=i+1)memory.mem[i]=(i*13+(i>>8)*7)&255;
  for(i=0;i<TOKENS;i=i+1)begin indices[i]=i%7;masks[i]=(i%3)!=0;end
  repeat(5)@(negedge clk);resetn=1;
  command("rectangle_4k_strides",0,'hff0,'h4ff8,3,37,208,192,0,0);
  for(r=0;r<3;r=r+1)for(c=0;c<37;c=c+1)begin
   actual_word=word_at('h4ff8+r*192+c*4);expected_word=word_at('hff0+r*208+c*4);
   if(actual_word!==expected_word)$fatal(1,"Rectangle mismatch r%0d c%0d actual%h expected%h",r,c,actual_word,expected_word);end
  indices[0]=6;indices[1]=0;indices[2]=3;
  command("embedding_gather",1,'h8000,'hc000,3,256,1024,1024,7,0);
  for(r=0;r<3;r=r+1)for(c=0;c<256;c=c+1)begin
   actual_word=word_at('hc000+r*1024+c*4);expected_word=word_at('h8000+indices[r]*1024+c*4);
   if(actual_word!==expected_word)$fatal(1,"Gather mismatch");end
  command("transpose_32x32_4k_strides",2,'h10ff0,'h180f8,32,32,192,160,0,0);
  for(r=0;r<32;r=r+1)for(c=0;c<32;c=c+1)begin
   actual_word=word_at('h180f8+c*160+r*4);expected_word=word_at('h10ff0+r*192+c*4);
   if(actual_word!==expected_word)$fatal(1,"Transpose mismatch %0d %0d",r,c);end
  command("mask_expand",3,0,'h220fc,2,TOKENS,0,TOKENS*4+32,0,0);
  for(r=0;r<2;r=r+1)for(c=0;c<TOKENS;c=c+1)
   if(word_at('h220fc+r*(TOKENS*4+32)+c*4)!==(masks[c]==1 ? 32'h80000000:32'hff7fffff))$fatal(1,"Mask mismatch");
  stream_count=0;stream_expected=TOKENS==256 ? 70 : TOKENS;stream_start='h2fff0;check_stream=1;
  command("read_words_backpressure_4k",4,'h2fff0,0,1,stream_expected,0,0,0,0);
  check_stream=0;if(stream_count!=stream_expected)$fatal(1,"Wrong streamed count");
  command("invalid_shape",0,0,'h4000,1,0,64,64,0,1);
  // These exact range boundaries exercise the newly registered validation path.
  command("source_range_overflow",0,0,'h4000,1,16,64,64,0,1,64'hfffffffffffffff0,BASE);
  command("destination_range_overflow",0,0,0,1,16,64,64,0,1,BASE,64'hfffffffffffffff0);
  command("overlapping_ranges_rejected",0,0,4,1,16,64,64,0,1);
  command("adjacent_ranges_permitted",0,0,64,1,16,64,64,0,0);
  for(c=0;c<16;c=c+1)begin
   actual_word=word_at(64+c*4);expected_word=word_at(c*4);
   if(actual_word!==expected_word)$fatal(1,"Adjacent range copy mismatch c=%0d actual=%h expected=%h",c,actual_word,expected_word);end
  indices[2]=7;command("invalid_index",1,'h8000,'hc000,3,256,1024,1024,7,2);indices[2]=3;
  masks[TOKENS-1]=2;command("invalid_mask",3,0,'h220fc,2,TOKENS,0,TOKENS*4+32,0,2);masks[TOKENS-1]=0;
  fault_mode=1;command("read_slverr_drain",0,0,'h4000,1,16,64,64,0,3);
  fault_mode=2;command("write_slverr",0,0,'h4000,1,16,64,64,0,4);
  fault_mode=3;command("early_rlast",0,0,'h4000,1,16,64,64,0,5);
  fault_mode=4;command("late_rlast_drain",0,0,'h4000,1,16,64,64,0,5);
  fault_mode=0;command("recovery_after_errors",0,0,'h4000,1,16,64,64,0,0);
  for(c=0;c<16;c=c+1)begin actual_word=word_at('h4000+c*4);expected_word=word_at(c*4);
   if(actual_word!==expected_word)$fatal(1,"Recovery mismatch");end
  if(TOKENS!=256)begin
   command("shape_rectangle_all_rows",0,'h40000,'h60000,TOKENS,64,256,256,0,0);
   for(r=0;r<TOKENS;r=r+1)for(c=0;c<64;c=c+1)begin
    actual_word=word_at('h60000+r*256+c*4);expected_word=word_at('h40000+r*256+c*4);
    if(actual_word!==expected_word)$fatal(1,"Full-length rectangle mismatch r=%0d c=%0d actual=%h expected=%h",r,c,actual_word,expected_word);end
   command("shape_transpose_all_rows",2,'h40000,'hc0000,TOKENS,64,256,TOKENS*4,0,0);
   for(r=0;r<TOKENS;r=r+1)for(c=0;c<64;c=c+1)begin
    actual_word=word_at('hc0000+c*TOKENS*4+r*4);expected_word=word_at('h40000+r*256+c*4);
    if(actual_word!==expected_word)$fatal(1,"Full-length transpose mismatch r=%0d c=%0d actual=%h expected=%h",r,c,actual_word,expected_word);end
   command("shape_embedding_all_indices",1,'h8000,'h100000,TOKENS,256,1024,1024,7,0);
   for(r=0;r<TOKENS;r=r+1)for(c=0;c<256;c=c+1)begin
    actual_word=word_at('h100000+r*1024+c*4);expected_word=word_at('h8000+indices[r]*1024+c*4);
    if(actual_word!==expected_word)$fatal(1,"Full-length gather mismatch r=%0d c=%0d actual=%h expected=%h",r,c,actual_word,expected_word);end
   indices[TOKENS-1]=7;command("shape_last_index_validation",1,'h8000,'h100000,TOKENS,256,1024,1024,7,2);
  end
  if(checks!=(TOKENS==256 ? 17 : 21))$fatal(1,"Case count changed");
  $display("PASS gate_data_mover %0d cases",checks);$finish;
 end
 initial begin #100000000;$fatal(1,"Bounded simulation timeout");end
endmodule
