// Simulation-only bounded AXI memory. One outstanding burst, no UVM objects.
// Backpressure is legal; cycle counts characterize this fixture, never DDR.
module axi_memory #(parameter DW=32, SIZE=16384)(
 input wire clk, resetn,
 input wire ARVALID, output wire ARREADY, input wire [63:0] ARADDR,
 input wire [0:0] ARID, input wire [7:0] ARLEN, input wire [2:0] ARSIZE, input wire [1:0] ARBURST,
 output wire RVALID, input wire RREADY, output reg [DW-1:0] RDATA,
 output wire RLAST, output reg [0:0] RID, output wire [1:0] RRESP,
 input wire AWVALID, output wire AWREADY, input wire [63:0] AWADDR,
 input wire [0:0] AWID, input wire [7:0] AWLEN, input wire [2:0] AWSIZE, input wire [1:0] AWBURST,
 input wire WVALID, output wire WREADY, input wire [DW-1:0] WDATA,
 input wire [DW/8-1:0] WSTRB, input wire WLAST,
 output reg BVALID, input wire BREADY, output reg [0:0] BID, output wire [1:0] BRESP
);
 reg [7:0] mem[0:SIZE-1];
 reg rd_busy=0,wr_busy=0;
 reg [63:0] rd_addr,wr_addr;
 reg [8:0] rd_left,wr_left;
 reg [2:0] rd_size,wr_size;
 reg [1:0] rd_burst,wr_burst;
 integer j,k; longint unsigned read_beats=0,write_beats=0;
 initial begin BVALID=0;RID=0;BID=0; for(k=0;k<SIZE;k=k+1)mem[k]=0;end
 assign ARREADY=!rd_busy;
 assign RVALID=rd_busy;
 assign RLAST=(rd_left==1);
 assign RRESP=0;
 assign AWREADY=!wr_busy&&!BVALID;
 assign WREADY=wr_busy&&!BVALID;
 assign BRESP=0;
 always @* begin
  RDATA=0;
  if(rd_busy)for(integer b=0;b<DW/8;b=b+1)begin
   if(((rd_addr/(DW/8))*(DW/8)+b)<SIZE)RDATA[b*8+:8]=mem[(rd_addr/(DW/8))*(DW/8)+b];
  end
 end
 always @(posedge clk) begin
  if(!resetn)begin rd_busy<=0;wr_busy<=0;BVALID<=0;end
  else begin
   if(ARVALID&&ARREADY)begin
    if(ARADDR>=SIZE || ARSIZE>$clog2(DW/8) || ARBURST>1)$fatal(1,"Invalid AXI read address/size/burst");
    rd_addr<=ARADDR;rd_left<={1'b0,ARLEN}+1;rd_size<=ARSIZE;rd_burst<=ARBURST;RID<=ARID;rd_busy<=1;
   end
   if(RVALID&&RREADY)begin
    read_beats<=read_beats+1;
    if(rd_left==1)rd_busy<=0;
    else begin rd_left<=rd_left-1;if(rd_burst==1)rd_addr<=rd_addr+(64'd1<<rd_size);end
   end
   if(AWVALID&&AWREADY)begin
    if(AWADDR>=SIZE || AWSIZE>$clog2(DW/8) || AWBURST>1)$fatal(1,"Invalid AXI write address/size/burst");
    wr_addr<=AWADDR;wr_left<={1'b0,AWLEN}+1;wr_size<=AWSIZE;wr_burst<=AWBURST;BID<=AWID;wr_busy<=1;
   end
   if(WVALID&&WREADY)begin
    write_beats<=write_beats+1;
    for(j=0;j<DW/8;j=j+1)if(WSTRB[j])begin
     if((wr_addr/(DW/8))*(DW/8)+j>=SIZE)$fatal(1,"AXI write exceeds fixture memory");
     mem[(wr_addr/(DW/8))*(DW/8)+j]<=WDATA[j*8+:8];
    end
    if(WLAST!=(wr_left==1))$fatal(1,"WLAST mismatch");
    if(wr_left==1)begin wr_busy<=0;BVALID<=1;end
    else begin wr_left<=wr_left-1;if(wr_burst==1)wr_addr<=wr_addr+(64'd1<<wr_size);end
   end
   if(BVALID&&BREADY)BVALID<=0;
  end
 end
endmodule
