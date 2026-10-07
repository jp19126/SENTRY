`timescale 1ns/1ps
// Simulation-only independent byte memory with deterministic backpressure.
// No physical DDR/NoC claim. Fault injection is confined to focused validation.
module gate_mover_axi_memory #(
 parameter integer DEPTH=262144,
 parameter logic [63:0] BASE=64'h100000000,
 parameter bit STALL=1,
 parameter bit PIPELINED_READ=0 // with STALL=0: original arithmetic-port nominal timing
)(input logic clk,resetn,input logic [2:0] fault_mode,
 input logic [63:0] m_axi_araddr, input logic [7:0] m_axi_arlen,
 input logic [2:0] m_axi_arsize, input logic [1:0] m_axi_arburst,
 input logic m_axi_arvalid, output logic m_axi_arready,
 output logic [31:0] m_axi_rdata, output logic [1:0] m_axi_rresp,
 output logic m_axi_rlast, m_axi_rvalid, input logic m_axi_rready,
 input logic [63:0] m_axi_awaddr, input logic [7:0] m_axi_awlen,
 input logic [2:0] m_axi_awsize, input logic [1:0] m_axi_awburst,
 input logic m_axi_awvalid, output logic m_axi_awready,
 input logic [31:0] m_axi_wdata, input logic [3:0] m_axi_wstrb,
 input logic m_axi_wlast, m_axi_wvalid, output logic m_axi_wready,
 output logic [1:0] m_axi_bresp, output logic m_axi_bvalid,
 input logic m_axi_bready
);
 byte unsigned mem[0:DEPTH-1];
 longint unsigned cycles,read_bursts,read_beats,write_bursts,write_beats;
 logic read_active,write_active;
 logic [63:0] ra,wa;
 integer rn,ri,wn,wi,bdelay;
 logic held_ar,held_aw,held_w;
 logic [76:0] old_ar,old_aw;
 logic [36:0] old_w;
 function automatic integer offset(input logic [63:0] address);
  begin
   if(address<BASE || address>BASE+DEPTH-4) $fatal(1,"AXI memory bounds %h",address);
   offset=integer'(address-BASE);
  end
 endfunction
 assign m_axi_arready=resetn&&!read_active&&!m_axi_rvalid&&(!STALL||cycles%5!=0);
 assign m_axi_awready=resetn&&!write_active&&!m_axi_bvalid&&bdelay==0&&(!STALL||cycles%7!=0);
 assign m_axi_wready=resetn&&write_active&&(!STALL||cycles%4!=0);
 always @(posedge clk) begin : memory_step
  integer a,j,next_ri;
  logic [63:0] next_ra;
  if(!resetn) begin
   cycles<=0;read_bursts<=0;read_beats<=0;write_bursts<=0;write_beats<=0;
   read_active<=0;write_active<=0;ra<=0;wa<=0;rn<=0;ri<=0;wn<=0;wi<=0;bdelay<=0;
   m_axi_rvalid<=0;m_axi_rdata<=0;m_axi_rresp<=0;m_axi_rlast<=0;
   m_axi_bvalid<=0;m_axi_bresp<=0;held_ar<=0;held_aw<=0;held_w<=0;
  end else begin
   cycles<=cycles+1;
   if(held_ar && (!m_axi_arvalid || {m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst}!==old_ar)) $fatal(1,"AR changed under backpressure");
   if(held_aw && (!m_axi_awvalid || {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst}!==old_aw)) $fatal(1,"AW changed under backpressure");
   if(held_w && (!m_axi_wvalid || {m_axi_wdata,m_axi_wstrb,m_axi_wlast}!==old_w)) $fatal(1,"W changed under backpressure");
   held_ar<=m_axi_arvalid&&!m_axi_arready;old_ar<={m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst};
   held_aw<=m_axi_awvalid&&!m_axi_awready;old_aw<={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst};
   held_w<=m_axi_wvalid&&!m_axi_wready;old_w<={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
   if(m_axi_arvalid&&m_axi_arready) begin
    if(m_axi_arlen>15 || m_axi_arsize!=2 || m_axi_arburst!=1 || m_axi_araddr[1:0]!=0 || integer'(m_axi_araddr[11:0])+(integer'(m_axi_arlen)+1)*4>4096) $fatal(1,"Illegal AR burst");
    a=offset(m_axi_araddr);a=offset(m_axi_araddr+(integer'(m_axi_arlen)+1)*4-4);
    read_active<=1;ra<=m_axi_araddr;rn<=integer'(m_axi_arlen)+1;ri<=0;read_bursts<=read_bursts+1;
    if(PIPELINED_READ&&!STALL)begin
     // Register first R immediately on the AR acceptance edge, so it can be
     // accepted on the following edge, matching linear_axi_tb.sv.
     a=offset(m_axi_araddr);m_axi_rdata<={mem[a+3],mem[a+2],mem[a+1],mem[a]};
     m_axi_rvalid<=1;m_axi_rresp<=fault_mode==1 ? 2'b10 : 2'b00;
     m_axi_rlast<=fault_mode==3 ? 1'b1 : (fault_mode==4 ? 1'b0 : m_axi_arlen==0);
    end
   end
   if(m_axi_rvalid&&m_axi_rready) begin
    m_axi_rvalid<=0;read_beats<=read_beats+1;
    if(m_axi_rlast)read_active<=0;else begin ri<=ri+1;ra<=ra+4;end
   end
   // Replace an accepted non-final beat on the same edge when opted in.
   // A stalled VALID beat never changes; accepted LAST never triggers refill.
   if(read_active&&(!m_axi_rvalid||(PIPELINED_READ&&m_axi_rready&&!m_axi_rlast))&&(!STALL||cycles%4!=1)) begin
    next_ra=ra;next_ri=ri;
    if(m_axi_rvalid&&m_axi_rready)begin next_ra=ra+4;next_ri=ri+1;end
    a=offset(next_ra);m_axi_rdata<={mem[a+3],mem[a+2],mem[a+1],mem[a]};
    m_axi_rresp<=fault_mode==1&&next_ri==0 ? 2'b10 : 2'b00;
    m_axi_rlast<=fault_mode==3 ? next_ri==0 : (fault_mode==4 ? next_ri==rn : next_ri==rn-1);
    m_axi_rvalid<=1;
   end
   if(m_axi_awvalid&&m_axi_awready) begin
    if(m_axi_awlen>15 || m_axi_awsize!=2 || m_axi_awburst!=1 || m_axi_awaddr[1:0]!=0 || integer'(m_axi_awaddr[11:0])+(integer'(m_axi_awlen)+1)*4>4096) $fatal(1,"Illegal AW burst");
    a=offset(m_axi_awaddr);a=offset(m_axi_awaddr+(integer'(m_axi_awlen)+1)*4-4);
    write_active<=1;wa<=m_axi_awaddr;wn<=integer'(m_axi_awlen)+1;wi<=0;write_bursts<=write_bursts+1;
   end
   if(m_axi_wvalid&&m_axi_wready) begin
    if(m_axi_wlast!=(wi==wn-1))$fatal(1,"Incorrect WLAST");
    if(m_axi_wstrb!=4'b1111)$fatal(1,"Unexpected partial float word");
    a=offset(wa);
    for(j=0;j<4;j=j+1)if(m_axi_wstrb[j])mem[a+j]=m_axi_wdata[8*j+:8];
    write_beats<=write_beats+1;
    if(m_axi_wlast)begin
     write_active<=0;
     if(PIPELINED_READ&&!STALL)begin
      // Registered B immediately after accepted last W; no added B delay.
      bdelay<=0;m_axi_bvalid<=1;m_axi_bresp<=fault_mode==2 ? 2'b10 : 2'b00;
     end else bdelay<=STALL ? 3 : 1;
    end
    else begin wi<=wi+1;wa<=wa+4;end
   end
   if(bdelay>0)begin
    bdelay<=bdelay-1;
    if(bdelay==1)begin m_axi_bvalid<=1;m_axi_bresp<=fault_mode==2 ? 2'b10 : 2'b00;end
   end
   if(m_axi_bvalid&&m_axi_bready)m_axi_bvalid<=0;
  end
 end
endmodule
