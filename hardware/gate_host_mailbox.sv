// Study-specific AXI-Lite mailbox. Full-word writes, independent AW/W channels,
// immutable pending request fields, captured completion before host acknowledgement.
module gate_host_mailbox #(parameter integer TOKENS=256) (
 (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER="XIL_INTERFACENAME clk, FREQ_HZ 200000000, ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET resetn" *) input wire clk,
 (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 resetn RST", X_INTERFACE_PARAMETER="XIL_INTERFACENAME resetn, POLARITY ACTIVE_LOW" *) input wire resetn,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *) input wire [7:0] s_axi_awaddr,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input wire s_axi_awvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output wire s_axi_awready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WDATA" *) input wire [31:0] s_axi_wdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *) input wire [3:0] s_axi_wstrb,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input wire s_axi_wvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output wire s_axi_wready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BRESP" *) output reg [1:0] s_axi_bresp,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output reg s_axi_bvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input wire s_axi_bready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input wire [7:0] s_axi_araddr,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input wire s_axi_arvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output wire s_axi_arready,
 (* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RDATA" *) output reg [31:0] s_axi_rdata,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RRESP" *) output reg [1:0] s_axi_rresp,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output reg s_axi_rvalid,(* X_INTERFACE_INFO="xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input wire s_axi_rready,
 output reg config_valid,input wire config_ready,
 output wire [15:0] config_w8_mask,output wire [31:0] config_threshold_bits,
 output wire [63:0] config_arena_base,config_model_id,
 output reg window_valid,input wire window_ready,
 output wire [63:0] window_sample_id,window_document_id,window_arena_base,
 output wire [31:0] window_index,output wire window_last,output wire [$clog2(TOKENS):0] window_valid_length,
 input wire result_valid,output wire result_ready,input wire [7:0] result_status,
 input wire [63:0] result_sample_id,result_document_id,result_model_id,
 input wire [31:0] result_window_index,result_risk_bits,result_max_bits,
 input wire result_last,result_reject
);
 reg [31:0] args[0:31];reg [31:0] completed[0:15];
 reg aw_held,w_held;reg [7:0] aw;reg [31:0] wd;reg [3:0] ws;
 reg busy,configured,result_pending;
 assign s_axi_awready=resetn&&!aw_held&&!s_axi_bvalid;
 assign s_axi_wready=resetn&&!w_held&&!s_axi_bvalid;
 assign s_axi_arready=resetn&&!s_axi_rvalid;
 assign config_w8_mask=args[4][15:0];assign config_threshold_bits=args[5];
 assign config_arena_base={args[7],args[6]};assign config_model_id={args[9],args[8]};
 assign window_sample_id={args[13],args[12]};assign window_document_id={args[15],args[14]};
 assign window_arena_base=config_arena_base;assign window_index=args[16];
 assign window_valid_length=args[17][$clog2(TOKENS):0];assign window_last=args[18][0];
 assign result_ready=!result_pending;
 always @(posedge clk)begin
  if(!resetn)begin
   aw_held<=0;w_held<=0;aw<=0;wd<=0;ws<=0;
   s_axi_bvalid<=0;s_axi_bresp<=0;s_axi_rvalid<=0;s_axi_rresp<=0;s_axi_rdata<=0;
   config_valid<=0;window_valid<=0;busy<=0;configured<=0;result_pending<=0;
   for(integer i=0;i<32;i=i+1)args[i]<=0;
   for(integer i=0;i<16;i=i+1)completed[i]<=0;
  end else begin
   if(config_valid&&config_ready)begin config_valid<=0;configured<=1;end
   if(window_valid&&window_ready)window_valid<=0;
   if(result_valid&&result_ready)begin
    result_pending<=1;busy<=0;
    completed[0]<={24'd0,result_status};completed[1]<=result_risk_bits;completed[2]<=result_max_bits;
    completed[3]<={30'd0,result_reject,result_last};
    completed[4]<=result_sample_id[31:0];completed[5]<=result_sample_id[63:32];
    completed[6]<=result_document_id[31:0];completed[7]<=result_document_id[63:32];
    completed[8]<=result_window_index;completed[9]<=result_model_id[31:0];completed[10]<=result_model_id[63:32];
   end
   if(s_axi_awvalid&&s_axi_awready)begin aw<=s_axi_awaddr;aw_held<=1;end
   if(s_axi_wvalid&&s_axi_wready)begin wd<=s_axi_wdata;ws<=s_axi_wstrb;w_held<=1;end
   if(s_axi_bvalid&&s_axi_bready)s_axi_bvalid<=0;
   if(aw_held&&w_held&&!s_axi_bvalid)begin
    aw_held<=0;w_held<=0;s_axi_bvalid<=1;s_axi_bresp<=0;
    if(aw[1:0]!=0 || ws!=4'hf)s_axi_bresp<=2;
    else if(aw==0)begin
     case(wd)
      1:if(config_ready&&!configured&&!busy&&!result_pending&&!window_valid)config_valid<=1;else s_axi_bresp<=2;
      2:if(configured&&window_ready&&!busy&&!result_pending&&!config_valid)begin window_valid<=1;busy<=1;end else s_axi_bresp<=2;
      4:if(result_pending)result_pending<=0;else s_axi_bresp<=2;
      default:s_axi_bresp<=2;
     endcase
    end else if(aw<8'h10 || aw>=8'h4c || busy || result_pending || config_valid || window_valid)s_axi_bresp<=2;
    else if(configured&&aw<8'h30)s_axi_bresp<=2;
    else args[aw[6:2]]<=wd;
   end
   if(s_axi_rvalid&&s_axi_rready)s_axi_rvalid<=0;
   if(s_axi_arvalid&&s_axi_arready)begin
    s_axi_rvalid<=1;s_axi_rresp<=0;
    if(s_axi_araddr[1:0]!=0)begin s_axi_rresp<=2;s_axi_rdata<=0;end
    else if(s_axi_araddr==4)s_axi_rdata<={27'd0,configured,busy,result_pending,window_ready,config_ready};
    else if(s_axi_araddr>=8'h80 && s_axi_araddr<8'hc0)s_axi_rdata<=completed[s_axi_araddr[5:2]];
    else if(s_axi_araddr>=8'h10 && s_axi_araddr<8'h4c)s_axi_rdata<=args[s_axi_araddr[6:2]];
    else begin s_axi_rresp<=2;s_axi_rdata<=0;end
   end
  end
 end
endmodule
