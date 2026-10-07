// Bounded verification-only harness. Compile linear_axi_tb.sv for the shared
// gate_axi_bus and gate_bounded_axi_memory definitions; select THIS top only.
// No UVM, shortreal, behavioral floating arithmetic, or DUT modification.
// Raw output bytes are checked independently by run_fixed_fp32_rtl.py.
`timescale 1ns/1ps
module fixed_fp32_axi_tb;
    logic clk=0, resetn=0;
    real clock_ns=5.0;
    integer op, rows, width, outputs, dot_bias, output_count;
    logic [31:0] scale_bits;
    string case_dir;
    longint unsigned timeout_cycles=10000000, cycle_count=0;
    longint unsigned start_cycle[0:1], done_cycle[0:1];
    integer starts=0, dones=0;
    bit active=0;
    integer x_extent;
    gate_axi_bus x_bus();
    gate_bounded_axi_memory #(.DEPTH(131072),.READ_ALLOWED(1),.WRITE_ALLOWED(0),.NAME("x"))
        x_memory(clk,resetn,x_extent,x_bus);
    integer y_extent;
    gate_axi_bus y_bus();
    gate_bounded_axi_memory #(.DEPTH(131072),.READ_ALLOWED(1),.WRITE_ALLOWED(0),.NAME("y"))
        y_memory(clk,resetn,y_extent,y_bus);
    integer z_extent;
    gate_axi_bus z_bus();
    gate_bounded_axi_memory #(.DEPTH(131072),.READ_ALLOWED(1),.WRITE_ALLOWED(0),.NAME("z"))
        z_memory(clk,resetn,z_extent,z_bus);
    integer accumulators_extent;
    gate_axi_bus accumulators_bus();
    gate_bounded_axi_memory #(.DEPTH(131072),.READ_ALLOWED(1),.WRITE_ALLOWED(0),.NAME("accumulators"))
        accumulators_memory(clk,resetn,accumulators_extent,accumulators_bus);
    integer output_extent;
    gate_axi_bus output_bus();
    gate_bounded_axi_memory #(.DEPTH(131072),.READ_ALLOWED(0),.WRITE_ALLOWED(1),.NAME("output"))
        output_memory(clk,resetn,output_extent,output_bus);
    integer codes_extent;
    gate_axi_bus codes_bus();
    gate_bounded_axi_memory #(.DEPTH(32768),.READ_ALLOWED(0),.WRITE_ALLOWED(1),.NAME("codes"))
        codes_memory(clk,resetn,codes_extent,codes_bus);
    logic c_awvalid=0,c_wvalid=0,c_bready=0,c_arvalid=0,c_rready=0;
    wire c_awready,c_wready,c_bvalid,c_arready,c_rvalid;
    logic [7:0] c_awaddr=0,c_araddr=0;
    logic [31:0] c_wdata=0;
    logic [3:0] c_wstrb=0;
    wire [31:0] c_rdata;
    wire [1:0] c_bresp,c_rresp;
    gate_fixed_fp32_top dut (
        .ap_clk(clk),
        .ap_rst_n(resetn),
        .m_axi_gmem_x_AWVALID(x_bus.awvalid),
        .m_axi_gmem_x_AWREADY(x_bus.awready),
        .m_axi_gmem_x_AWADDR(x_bus.awaddr),
        .m_axi_gmem_x_AWID(x_bus.awid),
        .m_axi_gmem_x_AWLEN(x_bus.awlen),
        .m_axi_gmem_x_AWSIZE(x_bus.awsize),
        .m_axi_gmem_x_AWBURST(x_bus.awburst),
        .m_axi_gmem_x_AWLOCK(x_bus.awlock),
        .m_axi_gmem_x_AWCACHE(),
        .m_axi_gmem_x_AWPROT(),
        .m_axi_gmem_x_AWQOS(),
        .m_axi_gmem_x_AWREGION(),
        .m_axi_gmem_x_AWUSER(),
        .m_axi_gmem_x_WVALID(x_bus.wvalid),
        .m_axi_gmem_x_WREADY(x_bus.wready),
        .m_axi_gmem_x_WDATA(x_bus.wdata),
        .m_axi_gmem_x_WSTRB(x_bus.wstrb),
        .m_axi_gmem_x_WLAST(x_bus.wlast),
        .m_axi_gmem_x_WID(x_bus.wid),
        .m_axi_gmem_x_WUSER(),
        .m_axi_gmem_x_ARVALID(x_bus.arvalid),
        .m_axi_gmem_x_ARREADY(x_bus.arready),
        .m_axi_gmem_x_ARADDR(x_bus.araddr),
        .m_axi_gmem_x_ARID(x_bus.arid),
        .m_axi_gmem_x_ARLEN(x_bus.arlen),
        .m_axi_gmem_x_ARSIZE(x_bus.arsize),
        .m_axi_gmem_x_ARBURST(x_bus.arburst),
        .m_axi_gmem_x_ARLOCK(x_bus.arlock),
        .m_axi_gmem_x_ARCACHE(),
        .m_axi_gmem_x_ARPROT(),
        .m_axi_gmem_x_ARQOS(),
        .m_axi_gmem_x_ARREGION(),
        .m_axi_gmem_x_ARUSER(),
        .m_axi_gmem_x_RVALID(x_bus.rvalid),
        .m_axi_gmem_x_RREADY(x_bus.rready),
        .m_axi_gmem_x_RDATA(x_bus.rdata),
        .m_axi_gmem_x_RLAST(x_bus.rlast),
        .m_axi_gmem_x_RID(x_bus.rid),
        .m_axi_gmem_x_RUSER(1'b0),
        .m_axi_gmem_x_RRESP(x_bus.rresp),
        .m_axi_gmem_x_BVALID(x_bus.bvalid),
        .m_axi_gmem_x_BREADY(x_bus.bready),
        .m_axi_gmem_x_BRESP(x_bus.bresp),
        .m_axi_gmem_x_BID(x_bus.bid),
        .m_axi_gmem_x_BUSER(1'b0),
        .m_axi_gmem_y_AWVALID(y_bus.awvalid),
        .m_axi_gmem_y_AWREADY(y_bus.awready),
        .m_axi_gmem_y_AWADDR(y_bus.awaddr),
        .m_axi_gmem_y_AWID(y_bus.awid),
        .m_axi_gmem_y_AWLEN(y_bus.awlen),
        .m_axi_gmem_y_AWSIZE(y_bus.awsize),
        .m_axi_gmem_y_AWBURST(y_bus.awburst),
        .m_axi_gmem_y_AWLOCK(y_bus.awlock),
        .m_axi_gmem_y_AWCACHE(),
        .m_axi_gmem_y_AWPROT(),
        .m_axi_gmem_y_AWQOS(),
        .m_axi_gmem_y_AWREGION(),
        .m_axi_gmem_y_AWUSER(),
        .m_axi_gmem_y_WVALID(y_bus.wvalid),
        .m_axi_gmem_y_WREADY(y_bus.wready),
        .m_axi_gmem_y_WDATA(y_bus.wdata),
        .m_axi_gmem_y_WSTRB(y_bus.wstrb),
        .m_axi_gmem_y_WLAST(y_bus.wlast),
        .m_axi_gmem_y_WID(y_bus.wid),
        .m_axi_gmem_y_WUSER(),
        .m_axi_gmem_y_ARVALID(y_bus.arvalid),
        .m_axi_gmem_y_ARREADY(y_bus.arready),
        .m_axi_gmem_y_ARADDR(y_bus.araddr),
        .m_axi_gmem_y_ARID(y_bus.arid),
        .m_axi_gmem_y_ARLEN(y_bus.arlen),
        .m_axi_gmem_y_ARSIZE(y_bus.arsize),
        .m_axi_gmem_y_ARBURST(y_bus.arburst),
        .m_axi_gmem_y_ARLOCK(y_bus.arlock),
        .m_axi_gmem_y_ARCACHE(),
        .m_axi_gmem_y_ARPROT(),
        .m_axi_gmem_y_ARQOS(),
        .m_axi_gmem_y_ARREGION(),
        .m_axi_gmem_y_ARUSER(),
        .m_axi_gmem_y_RVALID(y_bus.rvalid),
        .m_axi_gmem_y_RREADY(y_bus.rready),
        .m_axi_gmem_y_RDATA(y_bus.rdata),
        .m_axi_gmem_y_RLAST(y_bus.rlast),
        .m_axi_gmem_y_RID(y_bus.rid),
        .m_axi_gmem_y_RUSER(1'b0),
        .m_axi_gmem_y_RRESP(y_bus.rresp),
        .m_axi_gmem_y_BVALID(y_bus.bvalid),
        .m_axi_gmem_y_BREADY(y_bus.bready),
        .m_axi_gmem_y_BRESP(y_bus.bresp),
        .m_axi_gmem_y_BID(y_bus.bid),
        .m_axi_gmem_y_BUSER(1'b0),
        .m_axi_gmem_z_AWVALID(z_bus.awvalid),
        .m_axi_gmem_z_AWREADY(z_bus.awready),
        .m_axi_gmem_z_AWADDR(z_bus.awaddr),
        .m_axi_gmem_z_AWID(z_bus.awid),
        .m_axi_gmem_z_AWLEN(z_bus.awlen),
        .m_axi_gmem_z_AWSIZE(z_bus.awsize),
        .m_axi_gmem_z_AWBURST(z_bus.awburst),
        .m_axi_gmem_z_AWLOCK(z_bus.awlock),
        .m_axi_gmem_z_AWCACHE(),
        .m_axi_gmem_z_AWPROT(),
        .m_axi_gmem_z_AWQOS(),
        .m_axi_gmem_z_AWREGION(),
        .m_axi_gmem_z_AWUSER(),
        .m_axi_gmem_z_WVALID(z_bus.wvalid),
        .m_axi_gmem_z_WREADY(z_bus.wready),
        .m_axi_gmem_z_WDATA(z_bus.wdata),
        .m_axi_gmem_z_WSTRB(z_bus.wstrb),
        .m_axi_gmem_z_WLAST(z_bus.wlast),
        .m_axi_gmem_z_WID(z_bus.wid),
        .m_axi_gmem_z_WUSER(),
        .m_axi_gmem_z_ARVALID(z_bus.arvalid),
        .m_axi_gmem_z_ARREADY(z_bus.arready),
        .m_axi_gmem_z_ARADDR(z_bus.araddr),
        .m_axi_gmem_z_ARID(z_bus.arid),
        .m_axi_gmem_z_ARLEN(z_bus.arlen),
        .m_axi_gmem_z_ARSIZE(z_bus.arsize),
        .m_axi_gmem_z_ARBURST(z_bus.arburst),
        .m_axi_gmem_z_ARLOCK(z_bus.arlock),
        .m_axi_gmem_z_ARCACHE(),
        .m_axi_gmem_z_ARPROT(),
        .m_axi_gmem_z_ARQOS(),
        .m_axi_gmem_z_ARREGION(),
        .m_axi_gmem_z_ARUSER(),
        .m_axi_gmem_z_RVALID(z_bus.rvalid),
        .m_axi_gmem_z_RREADY(z_bus.rready),
        .m_axi_gmem_z_RDATA(z_bus.rdata),
        .m_axi_gmem_z_RLAST(z_bus.rlast),
        .m_axi_gmem_z_RID(z_bus.rid),
        .m_axi_gmem_z_RUSER(1'b0),
        .m_axi_gmem_z_RRESP(z_bus.rresp),
        .m_axi_gmem_z_BVALID(z_bus.bvalid),
        .m_axi_gmem_z_BREADY(z_bus.bready),
        .m_axi_gmem_z_BRESP(z_bus.bresp),
        .m_axi_gmem_z_BID(z_bus.bid),
        .m_axi_gmem_z_BUSER(1'b0),
        .m_axi_gmem_accumulators_AWVALID(accumulators_bus.awvalid),
        .m_axi_gmem_accumulators_AWREADY(accumulators_bus.awready),
        .m_axi_gmem_accumulators_AWADDR(accumulators_bus.awaddr),
        .m_axi_gmem_accumulators_AWID(accumulators_bus.awid),
        .m_axi_gmem_accumulators_AWLEN(accumulators_bus.awlen),
        .m_axi_gmem_accumulators_AWSIZE(accumulators_bus.awsize),
        .m_axi_gmem_accumulators_AWBURST(accumulators_bus.awburst),
        .m_axi_gmem_accumulators_AWLOCK(accumulators_bus.awlock),
        .m_axi_gmem_accumulators_AWCACHE(),
        .m_axi_gmem_accumulators_AWPROT(),
        .m_axi_gmem_accumulators_AWQOS(),
        .m_axi_gmem_accumulators_AWREGION(),
        .m_axi_gmem_accumulators_AWUSER(),
        .m_axi_gmem_accumulators_WVALID(accumulators_bus.wvalid),
        .m_axi_gmem_accumulators_WREADY(accumulators_bus.wready),
        .m_axi_gmem_accumulators_WDATA(accumulators_bus.wdata),
        .m_axi_gmem_accumulators_WSTRB(accumulators_bus.wstrb),
        .m_axi_gmem_accumulators_WLAST(accumulators_bus.wlast),
        .m_axi_gmem_accumulators_WID(accumulators_bus.wid),
        .m_axi_gmem_accumulators_WUSER(),
        .m_axi_gmem_accumulators_ARVALID(accumulators_bus.arvalid),
        .m_axi_gmem_accumulators_ARREADY(accumulators_bus.arready),
        .m_axi_gmem_accumulators_ARADDR(accumulators_bus.araddr),
        .m_axi_gmem_accumulators_ARID(accumulators_bus.arid),
        .m_axi_gmem_accumulators_ARLEN(accumulators_bus.arlen),
        .m_axi_gmem_accumulators_ARSIZE(accumulators_bus.arsize),
        .m_axi_gmem_accumulators_ARBURST(accumulators_bus.arburst),
        .m_axi_gmem_accumulators_ARLOCK(accumulators_bus.arlock),
        .m_axi_gmem_accumulators_ARCACHE(),
        .m_axi_gmem_accumulators_ARPROT(),
        .m_axi_gmem_accumulators_ARQOS(),
        .m_axi_gmem_accumulators_ARREGION(),
        .m_axi_gmem_accumulators_ARUSER(),
        .m_axi_gmem_accumulators_RVALID(accumulators_bus.rvalid),
        .m_axi_gmem_accumulators_RREADY(accumulators_bus.rready),
        .m_axi_gmem_accumulators_RDATA(accumulators_bus.rdata),
        .m_axi_gmem_accumulators_RLAST(accumulators_bus.rlast),
        .m_axi_gmem_accumulators_RID(accumulators_bus.rid),
        .m_axi_gmem_accumulators_RUSER(1'b0),
        .m_axi_gmem_accumulators_RRESP(accumulators_bus.rresp),
        .m_axi_gmem_accumulators_BVALID(accumulators_bus.bvalid),
        .m_axi_gmem_accumulators_BREADY(accumulators_bus.bready),
        .m_axi_gmem_accumulators_BRESP(accumulators_bus.bresp),
        .m_axi_gmem_accumulators_BID(accumulators_bus.bid),
        .m_axi_gmem_accumulators_BUSER(1'b0),
        .m_axi_gmem_output_AWVALID(output_bus.awvalid),
        .m_axi_gmem_output_AWREADY(output_bus.awready),
        .m_axi_gmem_output_AWADDR(output_bus.awaddr),
        .m_axi_gmem_output_AWID(output_bus.awid),
        .m_axi_gmem_output_AWLEN(output_bus.awlen),
        .m_axi_gmem_output_AWSIZE(output_bus.awsize),
        .m_axi_gmem_output_AWBURST(output_bus.awburst),
        .m_axi_gmem_output_AWLOCK(output_bus.awlock),
        .m_axi_gmem_output_AWCACHE(),
        .m_axi_gmem_output_AWPROT(),
        .m_axi_gmem_output_AWQOS(),
        .m_axi_gmem_output_AWREGION(),
        .m_axi_gmem_output_AWUSER(),
        .m_axi_gmem_output_WVALID(output_bus.wvalid),
        .m_axi_gmem_output_WREADY(output_bus.wready),
        .m_axi_gmem_output_WDATA(output_bus.wdata),
        .m_axi_gmem_output_WSTRB(output_bus.wstrb),
        .m_axi_gmem_output_WLAST(output_bus.wlast),
        .m_axi_gmem_output_WID(output_bus.wid),
        .m_axi_gmem_output_WUSER(),
        .m_axi_gmem_output_ARVALID(output_bus.arvalid),
        .m_axi_gmem_output_ARREADY(output_bus.arready),
        .m_axi_gmem_output_ARADDR(output_bus.araddr),
        .m_axi_gmem_output_ARID(output_bus.arid),
        .m_axi_gmem_output_ARLEN(output_bus.arlen),
        .m_axi_gmem_output_ARSIZE(output_bus.arsize),
        .m_axi_gmem_output_ARBURST(output_bus.arburst),
        .m_axi_gmem_output_ARLOCK(output_bus.arlock),
        .m_axi_gmem_output_ARCACHE(),
        .m_axi_gmem_output_ARPROT(),
        .m_axi_gmem_output_ARQOS(),
        .m_axi_gmem_output_ARREGION(),
        .m_axi_gmem_output_ARUSER(),
        .m_axi_gmem_output_RVALID(output_bus.rvalid),
        .m_axi_gmem_output_RREADY(output_bus.rready),
        .m_axi_gmem_output_RDATA(output_bus.rdata),
        .m_axi_gmem_output_RLAST(output_bus.rlast),
        .m_axi_gmem_output_RID(output_bus.rid),
        .m_axi_gmem_output_RUSER(1'b0),
        .m_axi_gmem_output_RRESP(output_bus.rresp),
        .m_axi_gmem_output_BVALID(output_bus.bvalid),
        .m_axi_gmem_output_BREADY(output_bus.bready),
        .m_axi_gmem_output_BRESP(output_bus.bresp),
        .m_axi_gmem_output_BID(output_bus.bid),
        .m_axi_gmem_output_BUSER(1'b0),
        .m_axi_gmem_codes_AWVALID(codes_bus.awvalid),
        .m_axi_gmem_codes_AWREADY(codes_bus.awready),
        .m_axi_gmem_codes_AWADDR(codes_bus.awaddr),
        .m_axi_gmem_codes_AWID(codes_bus.awid),
        .m_axi_gmem_codes_AWLEN(codes_bus.awlen),
        .m_axi_gmem_codes_AWSIZE(codes_bus.awsize),
        .m_axi_gmem_codes_AWBURST(codes_bus.awburst),
        .m_axi_gmem_codes_AWLOCK(codes_bus.awlock),
        .m_axi_gmem_codes_AWCACHE(),
        .m_axi_gmem_codes_AWPROT(),
        .m_axi_gmem_codes_AWQOS(),
        .m_axi_gmem_codes_AWREGION(),
        .m_axi_gmem_codes_AWUSER(),
        .m_axi_gmem_codes_WVALID(codes_bus.wvalid),
        .m_axi_gmem_codes_WREADY(codes_bus.wready),
        .m_axi_gmem_codes_WDATA(codes_bus.wdata),
        .m_axi_gmem_codes_WSTRB(codes_bus.wstrb),
        .m_axi_gmem_codes_WLAST(codes_bus.wlast),
        .m_axi_gmem_codes_WID(codes_bus.wid),
        .m_axi_gmem_codes_WUSER(),
        .m_axi_gmem_codes_ARVALID(codes_bus.arvalid),
        .m_axi_gmem_codes_ARREADY(codes_bus.arready),
        .m_axi_gmem_codes_ARADDR(codes_bus.araddr),
        .m_axi_gmem_codes_ARID(codes_bus.arid),
        .m_axi_gmem_codes_ARLEN(codes_bus.arlen),
        .m_axi_gmem_codes_ARSIZE(codes_bus.arsize),
        .m_axi_gmem_codes_ARBURST(codes_bus.arburst),
        .m_axi_gmem_codes_ARLOCK(codes_bus.arlock),
        .m_axi_gmem_codes_ARCACHE(),
        .m_axi_gmem_codes_ARPROT(),
        .m_axi_gmem_codes_ARQOS(),
        .m_axi_gmem_codes_ARREGION(),
        .m_axi_gmem_codes_ARUSER(),
        .m_axi_gmem_codes_RVALID(codes_bus.rvalid),
        .m_axi_gmem_codes_RREADY(codes_bus.rready),
        .m_axi_gmem_codes_RDATA(codes_bus.rdata),
        .m_axi_gmem_codes_RLAST(codes_bus.rlast),
        .m_axi_gmem_codes_RID(codes_bus.rid),
        .m_axi_gmem_codes_RUSER(1'b0),
        .m_axi_gmem_codes_RRESP(codes_bus.rresp),
        .m_axi_gmem_codes_BVALID(codes_bus.bvalid),
        .m_axi_gmem_codes_BREADY(codes_bus.bready),
        .m_axi_gmem_codes_BRESP(codes_bus.bresp),
        .m_axi_gmem_codes_BID(codes_bus.bid),
        .m_axi_gmem_codes_BUSER(1'b0),
        .s_axi_control_AWVALID(c_awvalid),
        .s_axi_control_AWREADY(c_awready),
        .s_axi_control_AWADDR(c_awaddr),
        .s_axi_control_WVALID(c_wvalid),
        .s_axi_control_WREADY(c_wready),
        .s_axi_control_WDATA(c_wdata),
        .s_axi_control_WSTRB(c_wstrb),
        .s_axi_control_ARVALID(c_arvalid),
        .s_axi_control_ARREADY(c_arready),
        .s_axi_control_ARADDR(c_araddr),
        .s_axi_control_RVALID(c_rvalid),
        .s_axi_control_RREADY(c_rready),
        .s_axi_control_RDATA(c_rdata),
        .s_axi_control_RRESP(c_rresp),
        .s_axi_control_BVALID(c_bvalid),
        .s_axi_control_BREADY(c_bready),
        .s_axi_control_BRESP(c_bresp),
        .interrupt()
    );

    initial begin
        if ($value$plusargs("CLOCK_NS=%f",clock_ns)) begin end
        if (!(clock_ns>0.0 && clock_ns<=1000.0)) $fatal(1,"GATE_FP32_FAIL clock");
        forever #(clock_ns/2.0) clk=~clk;
    end
    // The actual generated top accepts ap_start in state1 at this edge.
    always @(posedge clk) begin
        if (resetn) begin
            cycle_count=cycle_count+1;
            if (cycle_count>=timeout_cycles) $fatal(1,"GATE_FP32_FAIL timeout");
            if (dut.ap_start && dut.ap_CS_fsm_state1) begin
                if (active || starts>=2) $fatal(1,"GATE_FP32_FAIL unexpected start");
                start_cycle[starts]=cycle_count; starts=starts+1; active=1;
            end
            if (dut.ap_done) begin
                if (!active || dones>=starts) $fatal(1,"GATE_FP32_FAIL unexpected done");
                done_cycle[dones]=cycle_count; dones=dones+1; active=0;
            end
        end
    end
    task automatic control_write(input logic[7:0] address,input logic[31:0] value);
        begin
            @(negedge clk); c_awaddr=address; c_awvalid=1;
            do @(posedge clk); while (!c_awready);
            @(negedge clk); c_awvalid=0; c_wdata=value; c_wstrb=4'hf; c_wvalid=1;
            do @(posedge clk); while (!c_wready);
            @(negedge clk); c_wvalid=0; c_bready=1;
            do @(posedge clk); while (!c_bvalid);
            if (c_bresp!==2'b00) $fatal(1,"GATE_FP32_FAIL AXIL write");
            @(negedge clk); c_bready=0;
        end
    endtask
    task automatic control_read(input logic[7:0] address,output logic[31:0] value);
        begin
            @(negedge clk); c_araddr=address; c_arvalid=1;
            do @(posedge clk); while (!c_arready);
            @(negedge clk); c_arvalid=0; c_rready=1;
            do @(posedge clk); while (!c_rvalid);
            if (c_rresp!==2'b00 || $isunknown(c_rdata)) $fatal(1,"GATE_FP32_FAIL AXIL read");
            value=c_rdata;
            @(negedge clk); c_rready=0;
        end
    endtask
    initial begin : run_case
        integer fd;
        logic[31:0] status_value,status_valid,control_value;
        longint unsigned before_count[0:5][0:5];
        if (!$value$plusargs("OP=%d",op) || !$value$plusargs("ROWS=%d",rows)
            || !$value$plusargs("WIDTH=%d",width) || !$value$plusargs("OUTPUTS=%d",outputs)
            || !$value$plusargs("BIAS=%d",dot_bias) || !$value$plusargs("SCALE=%h",scale_bits)
            || !$value$plusargs("COUNT=%d",output_count) || !$value$plusargs("CASE_DIR=%s",case_dir))
            $fatal(1,"GATE_FP32_FAIL missing case arguments");
        if ($value$plusargs("TIMEOUT_CYCLES=%d",timeout_cycles)) begin end
        if (op<0 || op>8 || rows<1 || width<1 || output_count<1 || output_count>32768)
            $fatal(1,"GATE_FP32_FAIL invalid case");
        if (!$value$plusargs("X_BYTES=%d",x_extent) || x_extent<1 || x_extent>131072)
            $fatal(1,"GATE_FP32_FAIL x extent");
        for(integer i=0;i<131072;i=i+1) begin
            x_memory.mem[i]=0; x_memory.written[i]=0;
        end
        if (!$value$plusargs("Y_BYTES=%d",y_extent) || y_extent<1 || y_extent>131072)
            $fatal(1,"GATE_FP32_FAIL y extent");
        for(integer i=0;i<131072;i=i+1) begin
            y_memory.mem[i]=0; y_memory.written[i]=0;
        end
        if (!$value$plusargs("Z_BYTES=%d",z_extent) || z_extent<1 || z_extent>131072)
            $fatal(1,"GATE_FP32_FAIL z extent");
        for(integer i=0;i<131072;i=i+1) begin
            z_memory.mem[i]=0; z_memory.written[i]=0;
        end
        if (!$value$plusargs("ACCUMULATORS_BYTES=%d",accumulators_extent) || accumulators_extent<1 || accumulators_extent>131072)
            $fatal(1,"GATE_FP32_FAIL accumulators extent");
        for(integer i=0;i<131072;i=i+1) begin
            accumulators_memory.mem[i]=0; accumulators_memory.written[i]=0;
        end
        if (!$value$plusargs("OUTPUT_BYTES=%d",output_extent) || output_extent<1 || output_extent>131072)
            $fatal(1,"GATE_FP32_FAIL output extent");
        for(integer i=0;i<131072;i=i+1) begin
            output_memory.mem[i]=0; output_memory.written[i]=0;
        end
        if (!$value$plusargs("CODES_BYTES=%d",codes_extent) || codes_extent<1 || codes_extent>32768)
            $fatal(1,"GATE_FP32_FAIL codes extent");
        for(integer i=0;i<32768;i=i+1) begin
            codes_memory.mem[i]=0; codes_memory.written[i]=0;
        end

        repeat(8) @(posedge clk);
        @(negedge clk); resetn=1;
        repeat(4) @(posedge clk);
        // Actual generated control map. Six independent bank base pointers=0.
        control_write(8'h10,op);
        control_write(8'h18,0); control_write(8'h1c,0);
        control_write(8'h24,0); control_write(8'h28,0);
        control_write(8'h30,0); control_write(8'h34,0);
        control_write(8'h3c,0); control_write(8'h40,0);
        control_write(8'h48,0); control_write(8'h4c,0);
        control_write(8'h54,0); control_write(8'h58,0);
        control_write(8'h60,rows); control_write(8'h68,width);
        control_write(8'h70,outputs); control_write(8'h78,scale_bits);
        control_write(8'h80,dot_bias);
        for(integer rep=0;rep<2;rep=rep+1) begin
            $readmemh($sformatf("%s/x_%0d.hex",case_dir,rep),x_memory.mem,0,x_extent-1);
            $readmemh($sformatf("%s/y_%0d.hex",case_dir,rep),y_memory.mem,0,y_extent-1);
            $readmemh($sformatf("%s/z_%0d.hex",case_dir,rep),z_memory.mem,0,z_extent-1);
            $readmemh($sformatf("%s/accumulators_%0d.hex",case_dir,rep),accumulators_memory.mem,0,accumulators_extent-1);
            for(integer i=0;i<output_extent;i=i+1) begin
                output_memory.mem[i]=8'ha5; output_memory.written[i]=0;
            end
            for(integer i=0;i<codes_extent;i=i+1) begin
                codes_memory.mem[i]=8'ha5; codes_memory.written[i]=0;
            end
            before_count[0][0]=x_memory.read_bursts;
            before_count[0][1]=x_memory.read_beats;
            before_count[0][2]=x_memory.read_bytes;
            before_count[0][3]=x_memory.write_bursts;
            before_count[0][4]=x_memory.write_beats;
            before_count[0][5]=x_memory.write_bytes;
            before_count[1][0]=y_memory.read_bursts;
            before_count[1][1]=y_memory.read_beats;
            before_count[1][2]=y_memory.read_bytes;
            before_count[1][3]=y_memory.write_bursts;
            before_count[1][4]=y_memory.write_beats;
            before_count[1][5]=y_memory.write_bytes;
            before_count[2][0]=z_memory.read_bursts;
            before_count[2][1]=z_memory.read_beats;
            before_count[2][2]=z_memory.read_bytes;
            before_count[2][3]=z_memory.write_bursts;
            before_count[2][4]=z_memory.write_beats;
            before_count[2][5]=z_memory.write_bytes;
            before_count[3][0]=accumulators_memory.read_bursts;
            before_count[3][1]=accumulators_memory.read_beats;
            before_count[3][2]=accumulators_memory.read_bytes;
            before_count[3][3]=accumulators_memory.write_bursts;
            before_count[3][4]=accumulators_memory.write_beats;
            before_count[3][5]=accumulators_memory.write_bytes;
            before_count[4][0]=output_memory.read_bursts;
            before_count[4][1]=output_memory.read_beats;
            before_count[4][2]=output_memory.read_bytes;
            before_count[4][3]=output_memory.write_bursts;
            before_count[4][4]=output_memory.write_beats;
            before_count[4][5]=output_memory.write_bytes;
            before_count[5][0]=codes_memory.read_bursts;
            before_count[5][1]=codes_memory.read_beats;
            before_count[5][2]=codes_memory.read_bytes;
            before_count[5][3]=codes_memory.write_bursts;
            before_count[5][4]=codes_memory.write_beats;
            before_count[5][5]=codes_memory.write_bytes;

            control_write(8'h00,1);
            wait(dones==rep+1);
            @(negedge clk);
            if (x_memory.read_active || x_memory.write_active || x_bus.bvalid)
                $fatal(1,"GATE_FP32_FAIL done before x completion");
            if (y_memory.read_active || y_memory.write_active || y_bus.bvalid)
                $fatal(1,"GATE_FP32_FAIL done before y completion");
            if (z_memory.read_active || z_memory.write_active || z_bus.bvalid)
                $fatal(1,"GATE_FP32_FAIL done before z completion");
            if (accumulators_memory.read_active || accumulators_memory.write_active || accumulators_bus.bvalid)
                $fatal(1,"GATE_FP32_FAIL done before accumulators completion");
            if (output_memory.read_active || output_memory.write_active || output_bus.bvalid)
                $fatal(1,"GATE_FP32_FAIL done before output completion");
            if (codes_memory.read_active || codes_memory.write_active || codes_bus.bvalid)
                $fatal(1,"GATE_FP32_FAIL done before codes completion");

            control_read(8'h8c,status_valid);
            control_read(8'h88,status_value);
            if (status_valid[0]!==1'b1 || status_value!==32'd0)
                $fatal(1,"GATE_FP32_FAIL invalid status valid=%h value=%h",status_valid,status_value);
            control_read(8'h00,control_value);
            if(op==1) begin
                if(output_memory.write_bytes!=before_count[4][5])
                    $fatal(1,"GATE_FP32_FAIL quantize wrote FP output");
                fd=$fopen($sformatf("%s/actual_%0d.hex",case_dir,rep),"w");
                if(fd==0) $fatal(1,"GATE_FP32_FAIL output file");
                for(integer i=0;i<output_count;i=i+1) begin
                    if(!codes_memory.written[i]) $fatal(1,"GATE_FP32_FAIL unwritten code byte");
                    $fdisplay(fd,"%02x",codes_memory.mem[i]);
                end
            end else begin
                if(codes_memory.write_bytes!=before_count[5][5])
                    $fatal(1,"GATE_FP32_FAIL FP mode wrote codes");
                fd=$fopen($sformatf("%s/actual_%0d.hex",case_dir,rep),"w");
                if(fd==0) $fatal(1,"GATE_FP32_FAIL output file");
                for(integer i=0;i<output_count*4;i=i+1) begin
                    if(!output_memory.written[i]) $fatal(1,"GATE_FP32_FAIL unwritten FP byte");
                    $fdisplay(fd,"%02x",output_memory.mem[i]);
                end
            end
            $fclose(fd);
            $display("GATE_FP32_TRAFFIC {\"index\":%0d,\"port\":\"x\",\"read_bursts\":%0d,\"read_beats\":%0d,\"read_bytes\":%0d,\"write_bursts\":%0d,\"write_beats\":%0d,\"write_bytes\":%0d}",rep,x_memory.read_bursts-before_count[0][0],x_memory.read_beats-before_count[0][1],x_memory.read_bytes-before_count[0][2],x_memory.write_bursts-before_count[0][3],x_memory.write_beats-before_count[0][4],x_memory.write_bytes-before_count[0][5]);
            $display("GATE_FP32_TRAFFIC {\"index\":%0d,\"port\":\"y\",\"read_bursts\":%0d,\"read_beats\":%0d,\"read_bytes\":%0d,\"write_bursts\":%0d,\"write_beats\":%0d,\"write_bytes\":%0d}",rep,y_memory.read_bursts-before_count[1][0],y_memory.read_beats-before_count[1][1],y_memory.read_bytes-before_count[1][2],y_memory.write_bursts-before_count[1][3],y_memory.write_beats-before_count[1][4],y_memory.write_bytes-before_count[1][5]);
            $display("GATE_FP32_TRAFFIC {\"index\":%0d,\"port\":\"z\",\"read_bursts\":%0d,\"read_beats\":%0d,\"read_bytes\":%0d,\"write_bursts\":%0d,\"write_beats\":%0d,\"write_bytes\":%0d}",rep,z_memory.read_bursts-before_count[2][0],z_memory.read_beats-before_count[2][1],z_memory.read_bytes-before_count[2][2],z_memory.write_bursts-before_count[2][3],z_memory.write_beats-before_count[2][4],z_memory.write_bytes-before_count[2][5]);
            $display("GATE_FP32_TRAFFIC {\"index\":%0d,\"port\":\"accumulators\",\"read_bursts\":%0d,\"read_beats\":%0d,\"read_bytes\":%0d,\"write_bursts\":%0d,\"write_beats\":%0d,\"write_bytes\":%0d}",rep,accumulators_memory.read_bursts-before_count[3][0],accumulators_memory.read_beats-before_count[3][1],accumulators_memory.read_bytes-before_count[3][2],accumulators_memory.write_bursts-before_count[3][3],accumulators_memory.write_beats-before_count[3][4],accumulators_memory.write_bytes-before_count[3][5]);
            $display("GATE_FP32_TRAFFIC {\"index\":%0d,\"port\":\"output\",\"read_bursts\":%0d,\"read_beats\":%0d,\"read_bytes\":%0d,\"write_bursts\":%0d,\"write_beats\":%0d,\"write_bytes\":%0d}",rep,output_memory.read_bursts-before_count[4][0],output_memory.read_beats-before_count[4][1],output_memory.read_bytes-before_count[4][2],output_memory.write_bursts-before_count[4][3],output_memory.write_beats-before_count[4][4],output_memory.write_bytes-before_count[4][5]);
            $display("GATE_FP32_TRAFFIC {\"index\":%0d,\"port\":\"codes\",\"read_bursts\":%0d,\"read_beats\":%0d,\"read_bytes\":%0d,\"write_bursts\":%0d,\"write_beats\":%0d,\"write_bytes\":%0d}",rep,codes_memory.read_bursts-before_count[5][0],codes_memory.read_beats-before_count[5][1],codes_memory.read_bytes-before_count[5][2],codes_memory.write_bursts-before_count[5][3],codes_memory.write_beats-before_count[5][4],codes_memory.write_bytes-before_count[5][5]);

            $display("GATE_FP32_TRANSACTION {\"index\":%0d,\"start_cycle\":%0d,\"done_cycle\":%0d,\"latency_cycles\":%0d,\"status\":0,\"status_valid\":true,\"all_output_bytes_written\":true}",rep,start_cycle[rep],done_cycle[rep],done_cycle[rep]-start_cycle[rep]);
        end
        if(starts!=2 || dones!=2 || active) $fatal(1,"GATE_FP32_FAIL transaction count");
        $display("GATE_FP32_CAPTURED {\"capture_complete\":true,\"op\":%0d,\"rows\":%0d,\"width\":%0d,\"outputs\":%0d,\"output_count\":%0d,\"repetitions\":2,\"clock_period_ns\":%0.3f,\"latency_cycles\":[%0d,%0d],\"interval_cycles\":%0d,\"total_execution_cycles\":%0d,\"interval_includes_axilite_restart\":true,\"numerical_comparison\":\"offline_required\"}",op,rows,width,outputs,output_count,clock_ns,done_cycle[0]-start_cycle[0],done_cycle[1]-start_cycle[1],start_cycle[1]-start_cycle[0],done_cycle[1]-start_cycle[0]);
        $finish;
    end
endmodule
