// Direct verification of existing HLS RTL; no vendor UVM or DUT modification.
// Memory policy: one outstanding burst per port, first read data / final write
// response registered one cycle after its handshake, then one beat per cycle
// when READY permits. No extra DDR delay, cache, or cross-port contention model.
// Thus these cycle observations are distinct from vendor-UVM or board timings.
`timescale 1ns/1ps

interface gate_axi_bus;
    logic awvalid, awready, awid;
    logic [63:0] awaddr;
    logic [7:0] awlen;
    logic [2:0] awsize;
    logic [1:0] awburst, awlock;
    logic wvalid, wready, wlast, wid;
    logic [31:0] wdata;
    logic [3:0] wstrb;
    logic bvalid, bready, bid;
    logic [1:0] bresp;
    logic arvalid, arready, arid;
    logic [63:0] araddr;
    logic [7:0] arlen;
    logic [2:0] arsize;
    logic [1:0] arburst, arlock;
    logic rvalid, rready, rlast, rid;
    logic [31:0] rdata;
    logic [1:0] rresp;
endinterface

module gate_bounded_axi_memory #(
    parameter integer DEPTH = 262144,
    parameter bit READ_ALLOWED = 1,
    parameter bit WRITE_ALLOWED = 0,
    parameter NAME = "memory"
)(input logic clk, input logic resetn,
  input integer logical_bytes, gate_axi_bus bus);
    byte unsigned mem [0:DEPTH-1];
    bit written [0:DEPTH-1];
    logic read_active, write_active;
    logic [63:0] read_addr, write_addr, read_start, write_start;
    logic [2:0] read_size, write_size;
    logic [1:0] read_burst, write_burst;
    logic [7:0] read_length, write_length;
    logic write_id;
    integer read_remaining, write_remaining;
    longint unsigned read_bursts, read_beats, read_bytes;
    longint unsigned write_bursts, write_beats, write_bytes;
    bit held_ar, held_aw, held_w;
    logic [79:0] previous_ar, previous_aw;
    logic [37:0] previous_w;

    assign bus.arready = resetn && READ_ALLOWED && !read_active;
    assign bus.awready = resetn && WRITE_ALLOWED && !write_active && !bus.bvalid;
    assign bus.wready = resetn && WRITE_ALLOWED && write_active && !bus.bvalid;
    assign bus.rresp = 2'b00;
    assign bus.bresp = 2'b00;

    function automatic logic [63:0] next_address(
        input logic [63:0] current, input logic [63:0] first,
        input logic [2:0] size, input logic [1:0] burst, input logic [7:0] length);
        longint unsigned step, span, boundary, candidate;
        begin
            step = 64'd1 << size;
            if (burst == 0) next_address = current; // FIXED
            else if (burst == 1) next_address = (current / step + 1) * step; // INCR
            else begin // WRAP: aligned start and legal length checked at acceptance.
                span = step * (longint'(length) + 1);
                boundary = (first / span) * span;
                candidate = current + step;
                next_address = candidate >= boundary + span ? boundary : candidate;
            end
        end
    endfunction

    task automatic check_burst(input logic [63:0] address, input logic [7:0] length,
        input logic [2:0] size, input logic [1:0] burst, input logic [1:0] lock);
        longint unsigned step, last_byte;
        integer beats;
        begin
            if ($isunknown({address,length,size,burst,lock}))
                $fatal(1,"GATE_RTL_FAIL %s unknown AXI address/control",NAME);
            if (size > 2 || burst == 3 || lock != 0)
                $fatal(1,"GATE_RTL_FAIL %s unsupported AXI size/burst/lock",NAME);
            if (logical_bytes <= 0 || logical_bytes > DEPTH || address >= logical_bytes)
                $fatal(1,"GATE_RTL_FAIL %s address out of logical bounds addr=%0d extent=%0d",NAME,address,logical_bytes);
            step = 64'd1 << size;
            beats = integer'(length) + 1;
            if (burst == 0 && beats > 16)
                $fatal(1,"GATE_RTL_FAIL %s FIXED burst exceeds16 beats",NAME);
            if (burst == 2 && ((beats != 2 && beats != 4 && beats != 8 && beats != 16)
                              || address % step != 0))
                $fatal(1,"GATE_RTL_FAIL %s invalid WRAP alignment/length",NAME);
            if (burst == 1) begin
                last_byte = (address / step) * step + step * beats - 1;
                if (last_byte >= logical_bytes || (address >> 12) != (last_byte >> 12))
                    $fatal(1,"GATE_RTL_FAIL %s INCR burst crosses extent or4KiB",NAME);
            end
        end
    endtask

    task automatic check_beat(input logic [63:0] address, input logic [2:0] size);
        longint unsigned step, last_byte;
        begin
            step = 64'd1 << size;
            last_byte = (address / step) * step + step - 1;
            if (address >= logical_bytes || last_byte >= logical_bytes || last_byte >= DEPTH)
                $fatal(1,"GATE_RTL_FAIL %s beat out of bounds addr=%0d",NAME,address);
        end
    endtask

    function automatic logic [31:0] read_word(input logic [63:0] address);
        longint unsigned base;
        begin
            base = (address >> 2) << 2;
            if (base + 3 >= DEPTH) $fatal(1,"GATE_RTL_FAIL %s bus word out of bounds",NAME);
            read_word = {mem[base+3],mem[base+2],mem[base+1],mem[base]};
        end
    endfunction

    // Fixed arrays are initialized by the parent once. No per-beat objects,
    // dynamic queues, transaction histories, file transfers or waveform logging.
    always @(posedge clk) begin : memory_step
        logic [63:0] next_addr;
        longint unsigned word_base, step, last_byte;
        logic [3:0] allowed_strobes;
        integer valid_bytes;
        if (!resetn) begin
            read_active <= 0; write_active <= 0;
            bus.rvalid <= 0; bus.rlast <= 0; bus.rid <= 0; bus.rdata <= 0;
            bus.bvalid <= 0; bus.bid <= 0;
            read_remaining <= 0; write_remaining <= 0;
            read_bursts <= 0; read_beats <= 0; read_bytes <= 0;
            write_bursts <= 0; write_beats <= 0; write_bytes <= 0;
            held_ar <= 0; held_aw <= 0; held_w <= 0;
        end else begin
            if ($isunknown({bus.arvalid,bus.awvalid,bus.wvalid}))
                $fatal(1,"GATE_RTL_FAIL %s unknown master VALID",NAME);
            if (!READ_ALLOWED && bus.arvalid)
                $fatal(1,"GATE_RTL_FAIL %s unexpected read direction",NAME);
            if (!WRITE_ALLOWED && (bus.awvalid || bus.wvalid))
                $fatal(1,"GATE_RTL_FAIL %s unexpected write direction",NAME);
            if (held_ar && (bus.arvalid !== 1'b1 ||
                    {bus.araddr,bus.arid,bus.arlen,bus.arsize,bus.arburst,bus.arlock} !== previous_ar))
                $fatal(1,"GATE_RTL_FAIL %s AR changed under backpressure",NAME);
            if (held_aw && (bus.awvalid !== 1'b1 ||
                    {bus.awaddr,bus.awid,bus.awlen,bus.awsize,bus.awburst,bus.awlock} !== previous_aw))
                $fatal(1,"GATE_RTL_FAIL %s AW changed under backpressure",NAME);
            if (held_w && (bus.wvalid !== 1'b1 || {bus.wdata,bus.wstrb,bus.wlast,bus.wid} !== previous_w))
                $fatal(1,"GATE_RTL_FAIL %s W changed under backpressure",NAME);
            held_ar <= bus.arvalid && !bus.arready;
            held_aw <= bus.awvalid && !bus.awready;
            held_w <= bus.wvalid && !bus.wready;
            previous_ar <= {bus.araddr,bus.arid,bus.arlen,bus.arsize,bus.arburst,bus.arlock};
            previous_aw <= {bus.awaddr,bus.awid,bus.awlen,bus.awsize,bus.awburst,bus.awlock};
            previous_w <= {bus.wdata,bus.wstrb,bus.wlast,bus.wid};

            if (bus.arvalid && bus.arready) begin
                check_burst(bus.araddr,bus.arlen,bus.arsize,bus.arburst,bus.arlock);
                check_beat(bus.araddr,bus.arsize);
                if ($isunknown(bus.arid)) $fatal(1,"GATE_RTL_FAIL %s unknown ARID",NAME);
                read_active <= 1;
                read_addr <= bus.araddr; read_start <= bus.araddr;
                read_size <= bus.arsize; read_burst <= bus.arburst; read_length <= bus.arlen;
                read_remaining <= integer'(bus.arlen) + 1;
                bus.rid <= bus.arid; bus.rdata <= read_word(bus.araddr);
                bus.rvalid <= 1; bus.rlast <= bus.arlen == 0;
                read_bursts <= read_bursts + 1;
            end else if (bus.rvalid && bus.rready) begin
                step = 64'd1 << read_size;
                read_beats <= read_beats + 1;
                read_bytes <= read_bytes + step - (read_addr % step);
                if (read_remaining == 1) begin
                    read_active <= 0; bus.rvalid <= 0; bus.rlast <= 0;
                    read_remaining <= 0;
                end else begin
                    next_addr = next_address(read_addr,read_start,read_size,read_burst,read_length);
                    check_beat(next_addr,read_size);
                    read_addr <= next_addr; read_remaining <= read_remaining - 1;
                    bus.rdata <= read_word(next_addr); bus.rlast <= read_remaining == 2;
                end
            end

            if (bus.awvalid && bus.awready) begin
                check_burst(bus.awaddr,bus.awlen,bus.awsize,bus.awburst,bus.awlock);
                if ($isunknown(bus.awid)) $fatal(1,"GATE_RTL_FAIL %s unknown AWID",NAME);
                write_active <= 1; write_addr <= bus.awaddr; write_start <= bus.awaddr;
                write_size <= bus.awsize; write_burst <= bus.awburst; write_length <= bus.awlen;
                write_remaining <= integer'(bus.awlen) + 1; write_id <= bus.awid;
                bus.bid <= bus.awid; write_bursts <= write_bursts + 1;
            end
            if (bus.wvalid && bus.wready) begin
                check_beat(write_addr,write_size);
                if ($isunknown({bus.wdata,bus.wstrb,bus.wlast,bus.wid}) || bus.wid != write_id)
                    $fatal(1,"GATE_RTL_FAIL %s unknown write payload or mismatched WID",NAME);
                if (bus.wlast !== (write_remaining == 1))
                    $fatal(1,"GATE_RTL_FAIL %s WLAST does not match AWLEN",NAME);
                word_base = (write_addr >> 2) << 2;
                step = 64'd1 << write_size;
                last_byte = (write_addr / step) * step + step - 1;
                allowed_strobes = 0;
                for (integer lane=0; lane<4; lane=lane+1)
                    if (word_base+lane >= write_addr && word_base+lane <= last_byte)
                        allowed_strobes[lane] = 1;
                if ((bus.wstrb & ~allowed_strobes) != 0)
                    $fatal(1,"GATE_RTL_FAIL %s WSTRB outside transfer lanes",NAME);
                valid_bytes = 0;
                for (integer lane=0; lane<4; lane=lane+1) begin
                    if (bus.wstrb[lane]) begin
                        mem[word_base+lane] = bus.wdata[8*lane +: 8];
                        written[word_base+lane] = 1;
                        valid_bytes = valid_bytes + 1;
                    end
                end
                write_beats <= write_beats + 1; write_bytes <= write_bytes + valid_bytes;
                if (write_remaining == 1) begin
                    write_active <= 0; write_remaining <= 0; bus.bvalid <= 1;
                end else begin
                    write_addr <= next_address(write_addr,write_start,write_size,write_burst,write_length);
                    write_remaining <= write_remaining - 1;
                end
            end
            if (bus.bvalid && bus.bready) bus.bvalid <= 0;
        end
    end
endmodule

module linear_axi_tb;
    logic clk = 0, resetn = 0;
    real clock_ns = 5.0;
    integer rows, inner, outputs, bits, reps;
    integer a_extent, w_extent, o_extent;
    longint unsigned timeout_cycles = 50000000;
    longint unsigned cycle_count = 0;
    integer starts = 0, dones = 0;
    bit active = 0;
    longint unsigned start_cycle[0:1], done_cycle[0:1];
    longint signed expected[0:262143];
    gate_axi_bus a_bus();
    gate_axi_bus w_bus();
    gate_axi_bus o_bus();
    gate_bounded_axi_memory #(.DEPTH(262144),.READ_ALLOWED(1),.WRITE_ALLOWED(0),.NAME("a"))
        a_memory(clk,resetn,a_extent,a_bus);
    gate_bounded_axi_memory #(.DEPTH(262144),.READ_ALLOWED(1),.WRITE_ALLOWED(0),.NAME("w"))
        w_memory(clk,resetn,w_extent,w_bus);
    gate_bounded_axi_memory #(.DEPTH(1048576),.READ_ALLOWED(0),.WRITE_ALLOWED(1),.NAME("o"))
        o_memory(clk,resetn,o_extent,o_bus);

    logic c_awvalid=0, c_wvalid=0, c_bready=0, c_arvalid=0, c_rready=0;
    wire c_awready,c_wready,c_bvalid,c_arready,c_rvalid;
    logic [6:0] c_awaddr=0,c_araddr=0;
    logic [31:0] c_wdata=0;
    logic [3:0] c_wstrb=0;
    wire [31:0] c_rdata;
    wire [1:0] c_bresp,c_rresp;
    wire interrupt;

    gate_linear_top dut (
        .ap_clk(clk), .ap_rst_n(resetn),
        .m_axi_gmem_a_AWVALID(a_bus.awvalid),
        .m_axi_gmem_a_AWREADY(a_bus.awready),
        .m_axi_gmem_a_AWADDR(a_bus.awaddr),
        .m_axi_gmem_a_AWID(a_bus.awid),
        .m_axi_gmem_a_AWLEN(a_bus.awlen),
        .m_axi_gmem_a_AWSIZE(a_bus.awsize),
        .m_axi_gmem_a_AWBURST(a_bus.awburst),
        .m_axi_gmem_a_AWLOCK(a_bus.awlock),
        .m_axi_gmem_a_WVALID(a_bus.wvalid),
        .m_axi_gmem_a_WREADY(a_bus.wready),
        .m_axi_gmem_a_WDATA(a_bus.wdata),
        .m_axi_gmem_a_WSTRB(a_bus.wstrb),
        .m_axi_gmem_a_WLAST(a_bus.wlast),
        .m_axi_gmem_a_WID(a_bus.wid),
        .m_axi_gmem_a_ARVALID(a_bus.arvalid),
        .m_axi_gmem_a_ARREADY(a_bus.arready),
        .m_axi_gmem_a_ARADDR(a_bus.araddr),
        .m_axi_gmem_a_ARID(a_bus.arid),
        .m_axi_gmem_a_ARLEN(a_bus.arlen),
        .m_axi_gmem_a_ARSIZE(a_bus.arsize),
        .m_axi_gmem_a_ARBURST(a_bus.arburst),
        .m_axi_gmem_a_ARLOCK(a_bus.arlock),
        .m_axi_gmem_a_RVALID(a_bus.rvalid),
        .m_axi_gmem_a_RREADY(a_bus.rready),
        .m_axi_gmem_a_RDATA(a_bus.rdata),
        .m_axi_gmem_a_RLAST(a_bus.rlast),
        .m_axi_gmem_a_RID(a_bus.rid),
        .m_axi_gmem_a_RRESP(a_bus.rresp),
        .m_axi_gmem_a_BVALID(a_bus.bvalid),
        .m_axi_gmem_a_BREADY(a_bus.bready),
        .m_axi_gmem_a_BRESP(a_bus.bresp),
        .m_axi_gmem_a_BID(a_bus.bid),
        .m_axi_gmem_a_RUSER(1'b0),
        .m_axi_gmem_a_BUSER(1'b0),
        .m_axi_gmem_w_AWVALID(w_bus.awvalid),
        .m_axi_gmem_w_AWREADY(w_bus.awready),
        .m_axi_gmem_w_AWADDR(w_bus.awaddr),
        .m_axi_gmem_w_AWID(w_bus.awid),
        .m_axi_gmem_w_AWLEN(w_bus.awlen),
        .m_axi_gmem_w_AWSIZE(w_bus.awsize),
        .m_axi_gmem_w_AWBURST(w_bus.awburst),
        .m_axi_gmem_w_AWLOCK(w_bus.awlock),
        .m_axi_gmem_w_WVALID(w_bus.wvalid),
        .m_axi_gmem_w_WREADY(w_bus.wready),
        .m_axi_gmem_w_WDATA(w_bus.wdata),
        .m_axi_gmem_w_WSTRB(w_bus.wstrb),
        .m_axi_gmem_w_WLAST(w_bus.wlast),
        .m_axi_gmem_w_WID(w_bus.wid),
        .m_axi_gmem_w_ARVALID(w_bus.arvalid),
        .m_axi_gmem_w_ARREADY(w_bus.arready),
        .m_axi_gmem_w_ARADDR(w_bus.araddr),
        .m_axi_gmem_w_ARID(w_bus.arid),
        .m_axi_gmem_w_ARLEN(w_bus.arlen),
        .m_axi_gmem_w_ARSIZE(w_bus.arsize),
        .m_axi_gmem_w_ARBURST(w_bus.arburst),
        .m_axi_gmem_w_ARLOCK(w_bus.arlock),
        .m_axi_gmem_w_RVALID(w_bus.rvalid),
        .m_axi_gmem_w_RREADY(w_bus.rready),
        .m_axi_gmem_w_RDATA(w_bus.rdata),
        .m_axi_gmem_w_RLAST(w_bus.rlast),
        .m_axi_gmem_w_RID(w_bus.rid),
        .m_axi_gmem_w_RRESP(w_bus.rresp),
        .m_axi_gmem_w_BVALID(w_bus.bvalid),
        .m_axi_gmem_w_BREADY(w_bus.bready),
        .m_axi_gmem_w_BRESP(w_bus.bresp),
        .m_axi_gmem_w_BID(w_bus.bid),
        .m_axi_gmem_w_RUSER(1'b0),
        .m_axi_gmem_w_BUSER(1'b0),
        .m_axi_gmem_o_AWVALID(o_bus.awvalid),
        .m_axi_gmem_o_AWREADY(o_bus.awready),
        .m_axi_gmem_o_AWADDR(o_bus.awaddr),
        .m_axi_gmem_o_AWID(o_bus.awid),
        .m_axi_gmem_o_AWLEN(o_bus.awlen),
        .m_axi_gmem_o_AWSIZE(o_bus.awsize),
        .m_axi_gmem_o_AWBURST(o_bus.awburst),
        .m_axi_gmem_o_AWLOCK(o_bus.awlock),
        .m_axi_gmem_o_WVALID(o_bus.wvalid),
        .m_axi_gmem_o_WREADY(o_bus.wready),
        .m_axi_gmem_o_WDATA(o_bus.wdata),
        .m_axi_gmem_o_WSTRB(o_bus.wstrb),
        .m_axi_gmem_o_WLAST(o_bus.wlast),
        .m_axi_gmem_o_WID(o_bus.wid),
        .m_axi_gmem_o_ARVALID(o_bus.arvalid),
        .m_axi_gmem_o_ARREADY(o_bus.arready),
        .m_axi_gmem_o_ARADDR(o_bus.araddr),
        .m_axi_gmem_o_ARID(o_bus.arid),
        .m_axi_gmem_o_ARLEN(o_bus.arlen),
        .m_axi_gmem_o_ARSIZE(o_bus.arsize),
        .m_axi_gmem_o_ARBURST(o_bus.arburst),
        .m_axi_gmem_o_ARLOCK(o_bus.arlock),
        .m_axi_gmem_o_RVALID(o_bus.rvalid),
        .m_axi_gmem_o_RREADY(o_bus.rready),
        .m_axi_gmem_o_RDATA(o_bus.rdata),
        .m_axi_gmem_o_RLAST(o_bus.rlast),
        .m_axi_gmem_o_RID(o_bus.rid),
        .m_axi_gmem_o_RRESP(o_bus.rresp),
        .m_axi_gmem_o_BVALID(o_bus.bvalid),
        .m_axi_gmem_o_BREADY(o_bus.bready),
        .m_axi_gmem_o_BRESP(o_bus.bresp),
        .m_axi_gmem_o_BID(o_bus.bid),
        .m_axi_gmem_o_RUSER(1'b0),
        .m_axi_gmem_o_BUSER(1'b0),
        .s_axi_control_AWVALID(c_awvalid),
        .s_axi_control_AWREADY(c_awready),
        .s_axi_control_AWADDR(c_awaddr),
        .s_axi_control_WVALID(c_wvalid),
        .s_axi_control_WREADY(c_wready),
        .s_axi_control_WDATA(c_wdata),
        .s_axi_control_WSTRB(c_wstrb),
        .s_axi_control_BVALID(c_bvalid),
        .s_axi_control_BREADY(c_bready),
        .s_axi_control_BRESP(c_bresp),
        .s_axi_control_ARVALID(c_arvalid),
        .s_axi_control_ARREADY(c_arready),
        .s_axi_control_ARADDR(c_araddr),
        .s_axi_control_RVALID(c_rvalid),
        .s_axi_control_RREADY(c_rready),
        .s_axi_control_RDATA(c_rdata),
        .s_axi_control_RRESP(c_rresp),
        .interrupt(interrupt)
    );

    initial begin
        if ($value$plusargs("CLOCK_NS=%f",clock_ns)) begin end
        if (!(clock_ns > 0.0 && clock_ns <= 1000.0)) $fatal(1,"GATE_RTL_FAIL invalid CLOCK_NS");
        forever #(clock_ns/2.0) clk = ~clk;
    end

    // Passive DUT start/done timestamps, never inferred from software polling.
    // The observed HLS top accepts start in its first FSM state on this edge.
    always @(posedge clk) begin
        if (resetn) begin
            cycle_count = cycle_count + 1;
            if (cycle_count >= timeout_cycles) $fatal(1,"GATE_RTL_FAIL timeout cycles=%0d starts=%0d dones=%0d",cycle_count,starts,dones);
            if (dut.ap_start && dut.ap_CS_fsm_state1) begin
                if (active || starts >= 2) $fatal(1,"GATE_RTL_FAIL unexpected additional DUT start");
                start_cycle[starts] = cycle_count;
                starts = starts + 1;
                active = 1;
            end
            if (dut.ap_done) begin
                if (!active || dones >= starts) $fatal(1,"GATE_RTL_FAIL done without active transaction");
                done_cycle[dones] = cycle_count;
                dones = dones + 1;
                active = 0;
            end
        end
    end

    task automatic control_write(input logic [6:0] address,input logic [31:0] value);
        begin
            @(negedge clk); c_awaddr=address; c_awvalid=1;
            do @(posedge clk); while (!c_awready);
            @(negedge clk); c_awvalid=0; c_wdata=value; c_wstrb=4'hf; c_wvalid=1;
            do @(posedge clk); while (!c_wready);
            @(negedge clk); c_wvalid=0; c_bready=1;
            do @(posedge clk); while (!c_bvalid);
            if (c_bresp !== 2'b00) $fatal(1,"GATE_RTL_FAIL AXI-Lite write response");
            @(negedge clk); c_bready=0;
        end
    endtask

    task automatic control_read(input logic [6:0] address,output logic [31:0] value);
        begin
            @(negedge clk); c_araddr=address; c_arvalid=1;
            do @(posedge clk); while (!c_arready);
            @(negedge clk); c_arvalid=0; c_rready=1;
            do @(posedge clk); while (!c_rvalid);
            if (c_rresp !== 2'b00 || $isunknown(c_rdata)) $fatal(1,"GATE_RTL_FAIL AXI-Lite read response");
            value=c_rdata;
            @(negedge clk); c_rready=0;
        end
    endtask

    function automatic integer activation_at(input integer index);
        case (index % 5)
            0: activation_at=-128; 1: activation_at=-1; 2: activation_at=0;
            3: activation_at=1; default: activation_at=127;
        endcase
    endfunction

    function automatic integer weight_at(input integer index,input integer width_bits);
        if (width_bits == 4) begin
            case ((index*3+1) % 5)
                0: weight_at=-8; 1: weight_at=-1; 2: weight_at=0;
                3: weight_at=1; default: weight_at=7;
            endcase
        end else begin
            case ((index*5+2) % 9)
                0: weight_at=-128; 1: weight_at=-127; 2: weight_at=-17;
                3: weight_at=-16; 4: weight_at=-1; 5: weight_at=0;
                6: weight_at=15; 7: weight_at=16; default: weight_at=127;
            endcase
        end
    endfunction

    initial begin : run_point
        integer value, index;
        logic [31:0] status_value, control_value, actual_bits;
        longint signed actual_value, sum;
        longint unsigned a_bursts_before,a_beats_before,a_bytes_before;
        longint unsigned w_bursts_before,w_beats_before,w_bytes_before;
        longint unsigned o_bursts_before,o_beats_before,o_bytes_before;
        longint unsigned latency0,latency1,interval,total_cycles;
        if (!$value$plusargs("ROWS=%d",rows) || !$value$plusargs("INNER=%d",inner)
            || !$value$plusargs("OUTPUTS=%d",outputs) || !$value$plusargs("BITS=%d",bits)
            || !$value$plusargs("REPS=%d",reps))
            $fatal(1,"GATE_RTL_FAIL required plusargs ROWS INNER OUTPUTS BITS REPS");
        if ($value$plusargs("TIMEOUT_CYCLES=%d",timeout_cycles)) begin end
        if (rows < 1 || rows > 256 || reps != 2 || (bits != 4 && bits != 8)
            || !((inner == 256 && (outputs == 256 || outputs == 1024)) || (inner == 1024 && outputs == 256))
            || timeout_cycles < 1)
            $fatal(1,"GATE_RTL_FAIL invalid point or timeout");
        a_extent=rows*inner; w_extent=outputs*inner*bits/8; o_extent=rows*outputs*4;
        for (integer i=0;i<262144;i=i+1) begin
            a_memory.mem[i]=0; a_memory.written[i]=0;
            w_memory.mem[i]=0; w_memory.written[i]=0;
        end
        for (integer i=0;i<1048576;i=i+1) begin
            o_memory.mem[i]=0; o_memory.written[i]=0;
        end
        for (integer i=0;i<a_extent;i=i+1) a_memory.mem[i]=activation_at(i);
        for (integer i=0;i<outputs*inner;i=i+1) begin
            value=weight_at(i,bits);
            if (bits == 8) w_memory.mem[i]=value;
            else if (i % 2 == 0) w_memory.mem[i/2][3:0]=value[3:0];
            else w_memory.mem[i/2][7:4]=value[3:0];
        end
        for (integer r=0;r<rows;r=r+1) begin
            for (integer o=0;o<outputs;o=o+1) begin
                sum=0;
                for (integer k=0;k<inner;k=k+1)
                    sum=sum+longint'(activation_at(r*inner+k))*longint'(weight_at(o*inner+k,bits));
                if (sum < -64'sd2147483648 || sum > 64'sd2147483647)
                    $fatal(1,"GATE_RTL_FAIL reference int32 overflow sum=%0d row=%0d output=%0d",sum,r,o);
                expected[r*outputs+o]=sum;
            end
        end
        $display("GATE_RTL_CONFIG rows=%0d inner=%0d outputs=%0d bits=%0d reps=%0d clock_ns=%0.3f memory=bounded_single_outstanding_registered_response no_extra_ddr_delay=1",rows,inner,outputs,bits,reps,clock_ns);
        repeat(8) @(posedge clk);
        @(negedge clk); resetn=1;
        repeat(4) @(posedge clk);
        // Register addresses are from the actual generated control_s_axi.v.
        // All independent memory-bank base pointers are zero, including high words.
        control_write(7'h10,0); control_write(7'h14,0);
        control_write(7'h1c,0); control_write(7'h20,0);
        control_write(7'h28,0); control_write(7'h2c,0);
        control_write(7'h34,rows); control_write(7'h3c,inner);
        control_write(7'h44,outputs); control_write(7'h4c,bits);
        for (integer repetition=0;repetition<2;repetition=repetition+1) begin
            for (integer i=0;i<o_extent;i=i+1) begin
                o_memory.mem[i]=8'ha5; o_memory.written[i]=0;
            end
            a_bursts_before=a_memory.read_bursts; a_beats_before=a_memory.read_beats; a_bytes_before=a_memory.read_bytes;
            w_bursts_before=w_memory.read_bursts; w_beats_before=w_memory.read_beats; w_bytes_before=w_memory.read_bytes;
            o_bursts_before=o_memory.write_bursts; o_beats_before=o_memory.write_beats; o_bytes_before=o_memory.write_bytes;
            control_write(7'h00,1); // Start, with auto-restart disabled.
            wait(dones == repetition+1);
            @(negedge clk);
            if (a_memory.read_active || w_memory.read_active || o_memory.write_active || o_bus.bvalid)
                $fatal(1,"GATE_RTL_FAIL DUT done before memory completion");
            control_read(7'h54,status_value);
            if (status_value !== 0) $fatal(1,"GATE_RTL_FAIL DUT status=%0d",status_value);
            control_read(7'h00,control_value); // Clear sticky done only after passive timestamp capture.
            for (integer r=0;r<rows;r=r+1) begin
                for (integer o=0;o<outputs;o=o+1) begin
                    index=4*(r*outputs+o);
                    for (integer lane=0;lane<4;lane=lane+1)
                        if (!o_memory.written[index+lane]) $fatal(1,"GATE_RTL_FAIL unwritten output byte=%0d",index+lane);
                    actual_bits={o_memory.mem[index+3],o_memory.mem[index+2],o_memory.mem[index+1],o_memory.mem[index]};
                    actual_value=$signed(actual_bits);
                    if (actual_value != expected[r*outputs+o])
                        $fatal(1,"GATE_RTL_FAIL mismatch repetition=%0d row=%0d output=%0d actual=%0d expected=%0d",repetition,r,o,actual_value,expected[r*outputs+o]);
                end
            end
            $display("GATE_RTL_TRANSACTION {\"index\":%0d,\"start_cycle\":%0d,\"done_cycle\":%0d,\"latency_cycles\":%0d,\"a_read_bursts\":%0d,\"a_read_beats\":%0d,\"a_read_bytes\":%0d,\"w_read_bursts\":%0d,\"w_read_beats\":%0d,\"w_read_bytes\":%0d,\"o_write_bursts\":%0d,\"o_write_beats\":%0d,\"o_write_bytes\":%0d,\"exact_output_passed\":true}",
                repetition,start_cycle[repetition],done_cycle[repetition],done_cycle[repetition]-start_cycle[repetition],
                a_memory.read_bursts-a_bursts_before,a_memory.read_beats-a_beats_before,a_memory.read_bytes-a_bytes_before,
                w_memory.read_bursts-w_bursts_before,w_memory.read_beats-w_beats_before,w_memory.read_bytes-w_bytes_before,
                o_memory.write_bursts-o_bursts_before,o_memory.write_beats-o_beats_before,o_memory.write_bytes-o_bytes_before);
        end
        if (starts != 2 || dones != 2 || active) $fatal(1,"GATE_RTL_FAIL incorrect transaction count");
        latency0=done_cycle[0]-start_cycle[0]; latency1=done_cycle[1]-start_cycle[1];
        interval=start_cycle[1]-start_cycle[0]; total_cycles=done_cycle[1]-start_cycle[0];
        $display("GATE_RTL_VERIFIED {\"passed\":true,\"verification_method\":\"direct_axi_rtl_bounded_memory\",\"rows\":%0d,\"inner\":%0d,\"outputs\":%0d,\"bits\":%0d,\"repetitions\":2,\"clock_period_ns\":%0.3f,\"latency_cycles\":[%0d,%0d],\"interval_cycles\":%0d,\"total_execution_cycles\":%0d,\"interval_includes_axilite_restart\":true,\"vendor_cosim_pass\":false,\"dut_rtl_modified\":false}",
            rows,inner,outputs,bits,clock_ns,latency0,latency1,interval,total_cycles);
        $finish;
    end
endmodule
