`timescale 1ns/1ps
// Fixed-study AXI32 mover; no arithmetic interpretation of float payloads.
// Commands/cache inputs must remain semantically stable until rsp handshake.
// Reset while active requires the system to reset its connected AXI fabric too.
module gate_data_mover(
 input logic clk, resetn,
 input logic cmd_valid, output logic cmd_ready,
 input logic [2:0] cmd_op,
 input logic [63:0] cmd_src, cmd_dst,
 input logic [15:0] cmd_rows, cmd_cols,
 input logic [31:0] cmd_src_stride, cmd_dst_stride, cmd_limit,
 output logic rsp_valid, input logic rsp_ready, output logic [3:0] rsp_status,
 output logic [7:0] index_addr, input logic [31:0] index_value,
 output logic [7:0] mask_addr, input logic [7:0] mask_value,
 output logic rd_valid, input logic rd_ready,
 output logic [31:0] rd_data, output logic rd_last,
 output logic [63:0] m_axi_araddr, output logic [7:0] m_axi_arlen,
 output logic [2:0] m_axi_arsize, output logic [1:0] m_axi_arburst,
 output logic m_axi_arvalid, input logic m_axi_arready,
 input logic [31:0] m_axi_rdata, input logic [1:0] m_axi_rresp,
 input logic m_axi_rlast, m_axi_rvalid, output logic m_axi_rready,
 output logic [63:0] m_axi_awaddr, output logic [7:0] m_axi_awlen,
 output logic [2:0] m_axi_awsize, output logic [1:0] m_axi_awburst,
 output logic m_axi_awvalid, input logic m_axi_awready,
 output logic [31:0] m_axi_wdata, output logic [3:0] m_axi_wstrb,
 output logic m_axi_wlast, m_axi_wvalid, input logic m_axi_wready,
 input logic [1:0] m_axi_bresp, input logic m_axi_bvalid,
 output logic m_axi_bready
);
 localparam [2:0] RECT=0,GATHER=1,TRANSPOSE16=2,MASK=3,READ_WORDS=4;
 typedef enum logic [4:0] {IDLE,VALIDATE,CHECK_INDEX,CHECK_MASK,COPY_PREP,
 COPY_WRITE,COPY_NEXT,T_READ_PREP,T_READ_NEXT,T_WRITE_PREP,T_WRITE_NEXT,
 MASK_PREP,MASK_NEXT,STREAM_PREP,STREAM_NEXT,R_ADDR,R_DATA,R_DRAIN,
 W_ADDR,W_DATA,W_RESP,DONE} state_t;
 state_t state, after_read, after_write;
 logic [2:0] op;
 logic [63:0] src,dst,ra,wa;
 logic [15:0] rows,cols,row,col,tr,tc,inner,segment;
 logic [31:0] ss,ds,limit;
 logic [8:0] check_pos;
 logic [3:0] error;
 logic [4:0] nbeats,beat;
 logic [7:0] tile_offset;
 logic [31:0] tile[0:255];
 logic [63:0] source_address,destination_address,source_span,destination_span;
 logic bad_command;
 integer i;
 function automatic logic [4:0] burst_size(input logic [63:0] a,b,
                                           input logic [15:0] left,
                                           input logic use_b);
   integer n, boundary;
   begin
    n=left>16 ? 16 : left;
    boundary=(4096-int'(a[11:0]))/4;
    if(n>boundary) n=boundary;
    if(use_b) begin boundary=(4096-int'(b[11:0]))/4; if(n>boundary)n=boundary; end
    burst_size=n;
   end
 endfunction
 task automatic launch_read(input logic [63:0] address,input logic [4:0] count,
                            input logic [7:0] offset,input state_t continuation);
  begin ra<=address;nbeats<=count;beat<=0;tile_offset<=offset;
   after_read<=continuation;state<=R_ADDR;end
 endtask
 task automatic launch_write(input logic [63:0] address,input logic [4:0] count,
                             input state_t continuation);
  begin wa<=address;nbeats<=count;beat<=0;after_write<=continuation;state<=W_ADDR;end
 endtask
 always_comb begin
  source_address=src+64'(row)*64'(ss)+64'(col)*4;
  if(op==GATHER) source_address=src+64'(index_value)*1024+64'(col)*4;
  destination_address=dst+64'(row)*64'(ds)+64'(col)*4;
  source_span=64'(rows-1)*64'(ss)+64'(cols)*4;
  destination_span=64'(rows-1)*64'(ds)+64'(cols)*4;
  if(op==GATHER) source_span=64'(limit)*1024;
  if(op==TRANSPOSE16) destination_span=64'(cols-1)*64'(ds)+64'(rows)*4;
  if(op==READ_WORDS) source_span=64'(cols)*4;
  if(op==MASK) source_span=0;
  bad_command=(op>READ_WORDS || rows==0 || rows>256 || cols==0 || cols>256);
  if(op!=MASK) bad_command |= (src[1:0]!=0 || src>64'hffffffffffffffff-source_span);
  if(op!=READ_WORDS) begin
   bad_command |= (dst[1:0]!=0 || ds[1:0]!=0 || dst>64'hffffffffffffffff-destination_span);
   bad_command |= (ds<(op==TRANSPOSE16 ? 32'(rows)*4 : 32'(cols)*4));
  end
  if(op==RECT || op==TRANSPOSE16) bad_command |= (ss[1:0]!=0 || ss<32'(cols)*4);
  if(op==GATHER) bad_command |= (cols!=256 || limit==0 || limit>30522);
  if(op==TRANSPOSE16) bad_command |= (rows[3:0]!=0 || cols[3:0]!=0);
  if(op==MASK) bad_command |= (cols!=256 || rows>128);
  if(op==READ_WORDS) bad_command |= (rows!=1);
  // In-place/overlapping memory transforms are outside this fixed schedule.
  if(op!=READ_WORDS && op!=MASK)
   bad_command |= (src<dst+destination_span && dst<src+source_span);
 end
 assign cmd_ready=state==IDLE;
 assign rsp_valid=state==DONE;
 assign rsp_status=error;
 assign index_addr=state==CHECK_INDEX ? check_pos[7:0] : row[7:0];
 assign mask_addr=state==CHECK_MASK ? check_pos[7:0] : (col[7:0]+{3'b0,beat});
 assign m_axi_araddr=ra; assign m_axi_arlen={3'b0,nbeats}-1;
 assign m_axi_arsize=3'd2; assign m_axi_arburst=2'b01;
 assign m_axi_arvalid=state==R_ADDR;
 assign m_axi_rready=(state==R_DRAIN) || (state==R_DATA && (op!=READ_WORDS || error!=0 || m_axi_rresp!=0 || rd_ready));
 assign rd_valid=state==R_DATA && op==READ_WORDS && error==0 && m_axi_rvalid && m_axi_rresp==0;
 assign rd_data=m_axi_rdata;
 assign rd_last=(col+beat==cols-1);
 assign m_axi_awaddr=wa;assign m_axi_awlen={3'b0,nbeats}-1;
 assign m_axi_awsize=3'd2;assign m_axi_awburst=2'b01;
 assign m_axi_awvalid=state==W_ADDR;
 assign m_axi_wvalid=state==W_DATA;
 assign m_axi_wstrb=4'b1111;
 assign m_axi_wlast=beat==nbeats-1;
 assign m_axi_bready=state==W_RESP;
 always_comb begin
  m_axi_wdata=tile[beat[3:0]];
  if(op==TRANSPOSE16) m_axi_wdata=tile[(segment+beat)*16+inner];
  if(op==MASK) m_axi_wdata=mask_value==1 ? 32'h80000000 : 32'hff7fffff;
 end
 always_ff @(posedge clk) begin
  if(!resetn) begin
   state<=IDLE;error<=0;op<=0;src<=0;dst<=0;rows<=0;cols<=0;ss<=0;ds<=0;limit<=0;
   ra<=0;wa<=0;nbeats<=0;beat<=0;row<=0;col<=0;tr<=0;tc<=0;inner<=0;segment<=0;
   tile_offset<=0;check_pos<=0;after_read<=IDLE;after_write<=IDLE;
  end else case(state)
   IDLE: if(cmd_valid) begin
    op<=cmd_op;src<=cmd_src;dst<=cmd_dst;rows<=cmd_rows;cols<=cmd_cols;
    ss<=cmd_src_stride;ds<=cmd_dst_stride;limit<=cmd_limit;error<=0;
    row<=0;col<=0;tr<=0;tc<=0;inner<=0;segment<=0;check_pos<=0;state<=VALIDATE;
   end
   VALIDATE: if(bad_command) begin error<=1;state<=DONE;end
    else if(op==GATHER) state<=CHECK_INDEX;
    else if(op==MASK) state<=CHECK_MASK;
    else if(op==TRANSPOSE16) state<=T_READ_PREP;
    else if(op==READ_WORDS) state<=STREAM_PREP;
    else state<=COPY_PREP;
   CHECK_INDEX: if(index_value>=limit) begin error<=2;state<=DONE;end
    else if(check_pos==rows-1) state<=COPY_PREP;else check_pos<=check_pos+1;
   CHECK_MASK: if(mask_value>1) begin error<=2;state<=DONE;end
    else if(check_pos==255) state<=MASK_PREP;else check_pos<=check_pos+1;
   COPY_PREP: launch_read(source_address,burst_size(source_address,destination_address,cols-col,1),0,COPY_WRITE);
   COPY_WRITE: launch_write(destination_address,nbeats,COPY_NEXT);
   COPY_NEXT: if(col+nbeats==cols) begin
     col<=0;if(row==rows-1)state<=DONE;else begin row<=row+1;state<=COPY_PREP;end
    end else begin col<=col+nbeats;state<=COPY_PREP;end
   T_READ_PREP: launch_read(src+64'(tr+inner)*64'(ss)+64'(tc+segment)*4,
     burst_size(src+64'(tr+inner)*64'(ss)+64'(tc+segment)*4,0,16-segment,0),
     8'(inner*16+segment),T_READ_NEXT);
   T_READ_NEXT: if(segment+nbeats==16) begin segment<=0;
     if(inner==15) begin inner<=0;state<=T_WRITE_PREP;end
     else begin inner<=inner+1;state<=T_READ_PREP;end
    end else begin segment<=segment+nbeats;state<=T_READ_PREP;end
   T_WRITE_PREP: launch_write(dst+64'(tc+inner)*64'(ds)+64'(tr+segment)*4,
     burst_size(dst+64'(tc+inner)*64'(ds)+64'(tr+segment)*4,0,16-segment,0),T_WRITE_NEXT);
   T_WRITE_NEXT: if(segment+nbeats==16) begin segment<=0;
     if(inner==15) begin inner<=0;
      if(tc+16==cols) begin tc<=0;
       if(tr+16==rows)state<=DONE;else begin tr<=tr+16;state<=T_READ_PREP;end
      end else begin tc<=tc+16;state<=T_READ_PREP;end
     end else begin inner<=inner+1;state<=T_WRITE_PREP;end
    end else begin segment<=segment+nbeats;state<=T_WRITE_PREP;end
   MASK_PREP: launch_write(destination_address,burst_size(destination_address,0,cols-col,0),MASK_NEXT);
   MASK_NEXT: if(col+nbeats==cols) begin col<=0;
     if(row==rows-1)state<=DONE;else begin row<=row+1;state<=MASK_PREP;end
    end else begin col<=col+nbeats;state<=MASK_PREP;end
   STREAM_PREP: launch_read(src+64'(col)*4,burst_size(src+64'(col)*4,0,cols-col,0),0,STREAM_NEXT);
   STREAM_NEXT: if(col+nbeats==cols)state<=DONE;else begin col<=col+nbeats;state<=STREAM_PREP;end
   R_ADDR: if(m_axi_arready) state<=R_DATA;
   R_DATA: if(m_axi_rvalid && m_axi_rready) begin
    if(op!=READ_WORDS) tile[tile_offset+beat]<=m_axi_rdata;
    if(m_axi_rresp!=0 && error==0) error<=3;
    if(m_axi_rlast && beat!=nbeats-1) begin error<=5;state<=DONE;end
    else if(beat==nbeats-1) begin
     if(!m_axi_rlast) begin error<=5;state<=R_DRAIN;end
     else if(error!=0 || m_axi_rresp!=0) state<=DONE;
     else state<=after_read;
    end else beat<=beat+1;
   end
   R_DRAIN: if(m_axi_rvalid && m_axi_rlast) state<=DONE;
   W_ADDR: if(m_axi_awready) state<=W_DATA;
   W_DATA: if(m_axi_wready) begin
    if(beat==nbeats-1)state<=W_RESP;else beat<=beat+1;
   end
   W_RESP: if(m_axi_bvalid) begin
    if(m_axi_bresp!=0) begin error<=4;state<=DONE;end else state<=after_write;
   end
   DONE: if(rsp_ready)state<=IDLE;
   default: begin error<=1;state<=DONE;end
  endcase
 end
endmodule
