"""Fixed L256 detector command/buffer plan; no hardware allocation or execution.

Only this BERT-Mini study is supported. Offsets describe a proposed 64 MiB DDR
arena. Generated AXI-Lite offsets are read from actual successful HLS builds.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re
import struct

from build_linear_hls import active_linear_profiles, active_linear_variants

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/goal4/internal_integration/command_plan.json"
REPORT = ROOT / "reports/goal4_command_plan.md"
L, D, H, HD, FF, CAP = 256, 256, 4, 64, 1024, 32768
ARENA = 64 * 1024 * 1024
OPS = dict(DOT=0, QUANTIZE=1, RESCALE_BIAS=2, EMBED_ADD=3,
           RESIDUAL_ADD=4, LAYER_NORM=5, SOFTMAX=6, GELU=7, TANH=8)
EXPECTED_CALLS = [165, 72, 72, 2, 16, 18, 33, 32, 1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def regs(path):
    values = {name: int(value, 16) for name, value in re.findall(
        r"\bADDR_([A-Z0-9_]+)\s*=\s*\d+'h([0-9a-fA-F]+)", path.read_text())}
    if any(values.get(name) != address for name, address in
           (("AP_CTRL", 0), ("GIE", 4), ("IER", 8), ("ISR", 12))):
        raise ValueError("Unexpected generated control protocol.")
    return values


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def exact_threshold_bits(value):
    """The calibrated observed-FP32 threshold must survive a host round trip."""
    bits = float_bits(value)
    restored = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not 0 <= value <= 1 or restored != value:
        raise ValueError("Threshold is not an exact finite FP32 value in [0,1].")
    return 0 if restored == 0 else bits


def write_rtl_plan(plan, folder):
    """One fixed 512-bit descriptor per planned command, not a general ISA."""
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for c in plan["commands"]:
        w = [0] * 16
        if c["kind"] == "service":
            a = c["arguments"]
            if c["engine"] == "integer":
                w[:8] = [1, a["ACTIVATIONS"]["arena_offset"], a["WEIGHTS"]["arena_offset"],
                         a["OUTPUT_R"]["arena_offset"], a["ROWS"], a["INNER"], a["OUTPUTS"],
                         a["WEIGHT_BITS"]["frozen_precision_group"]]
            else:
                y = a["Y"]
                group = y.get("select_by_frozen_precision_group", 16)
                y4 = y["alternatives"]["4"] if group < 16 else y
                y8 = y["alternatives"]["8"] if group < 16 else y
                w = [2, a["OP"], a["X"]["arena_offset"], y4["arena_offset"], a["Z"]["arena_offset"],
                     a["ACCUMULATORS"]["arena_offset"], a["OUTPUT_R"]["arena_offset"], a["CODES"]["arena_offset"],
                     a["ROWS"], a["WIDTH"], a["OUTPUTS"], a["SCALE"].get("f32_bits", 0),
                     a["SCALE"].get("local_input_scale_index", 255), a["DOT_BIAS"], group, y8["arena_offset"]]
        elif c["kind"] == "movement":
            op = {"rectangle": 0, "indexed_embedding_rows": 1, "transpose_f32_tiles16": 2,
                  "binary_mask_to_fp32_repeated_rows": 3}[c["mode"]]
            w[:10] = [3, op, c["source"]["arena_offset"] if isinstance(c["source"], dict) else 0,
                      c["destination"]["arena_offset"], c["rows"], c["columns"],
                      c.get("source_stride_bytes") or 0, c.get("destination_stride_bytes") or 1024,
                      c.get("index_limit", 0), int(c.get("index_source") == "local_type_ids")]
        elif c["name"] == "read_risk_and_update_document":
            w[:2] = [4, c["source"]["arena_offset"]]
        records.append("".join(f"{v:08x}" for v in reversed(w)))
    (folder / "command_plan.mem").write_text("\n".join(records) + "\n")
    constants = [f"localparam integer PLAN_COUNT = {len(records)};", f"localparam [63:0] ARENA_BYTES = 64'd{ARENA};"]
    for name in ("token_ids", "type_ids", "attention_mask", "input_scales"):
        constants.append(f"localparam [31:0] {name.upper()}_OFFSET = 32'd{plan['buffers'][name]['offset']};")
    (folder / "command_plan_constants.svh").write_text("\n".join(constants) + "\n")
    return len(records)


def create_plan(config):
    model = read(ROOT / "checkpoints/quantized/qat_w8_a8/config.json")
    if (model["vocab_size"], model["type_vocab_size"], model["max_position_embeddings"],
        model["hidden_size"], model["num_hidden_layers"], model["num_attention_heads"],
        model["intermediate_size"]) != (30522, 2, 512, D, 4, H, FF):
        raise ValueError("This plan is fixed to the observed BERT-Mini model.")
    variants = active_linear_variants(config)
    identities, integer_registers = [], None
    for profile in active_linear_profiles(config):
        name = profile["id"]
        build = ROOT / ("build/linear_hls_cached_static_bank" if variants[name] == "weight_cache_static_bank"
                        else "build/linear_hls_cached") / name
        run = read(build / "run.json")
        if not run.get("hls_synthesis_performed") or run.get("returncode") != 0 or run["kernel_variant"] != variants[name]:
            raise ValueError("Require the actual selected integer source build.")
        control = build / "project/solution/syn/verilog/gate_linear_top_control_s_axi.v"
        current = regs(control)
        if integer_registers is not None and integer_registers != current:
            raise ValueError("Common profiles have different generated register contracts.")
        integer_registers = current
        identities.append(dict(profile=name, kernel_variant=variants[name],
            source_revision=run.get("source_revision"), synthesis_recorded_utc=run["recorded_utc"],
            control_source=control.relative_to(ROOT).as_posix()))
    fp_build = ROOT / "build/fixed_fp32_hls"
    fp_run = read(fp_build / "run.json")
    fp_control = fp_build / "project/solution/syn/verilog/gate_fixed_fp32_top_control_s_axi.v"
    fp_registers = regs(fp_control)
    if not fp_run.get("synthesis_passed") or not fp_run.get("c_simulation_passed") or fp_run.get("returncode") != 0:
        raise ValueError("Require the successful common FP32 synthesis.")
    header = (fp_build / "fixed_fp32_service.hpp").read_text()
    if '#define GATE_FIXED_FP32_SCHEDULE_REVISION "panel_cache_batched_attention_v1"' not in header:
        raise ValueError("Common FP32 service revision differs.")

    buffers, commands, matrices = {}, [], []
    cursor = 0

    def allocate(name, size, dtype, shape, role):
        nonlocal cursor
        cursor = (cursor + 4095) // 4096 * 4096
        buffers[name] = dict(offset=cursor, bytes=size, dtype=dtype, shape=shape, role=role)
        cursor += size
        if cursor > ARENA:
            raise ValueError("Fixed plan exceeds the proposed arena.")
        return name

    def ref(name, offset=0, size=0):
        b = buffers[name]
        if offset < 0 or size < 0 or offset + size > b["bytes"]:
            raise ValueError("Out-of-buffer planned access: " + name)
        return dict(buffer=name, buffer_offset=offset, arena_offset=b["offset"] + offset, extent_bytes=size)

    def append(kind, name, **fields):
        command = dict(index=len(commands), kind=kind, name=name, **fields)
        commands.append(command)
        return command

    def move(mode, name, source, destination, rows, columns, element_bytes=4,
             source_stride=None, destination_stride=None, **extra):
        append("movement", name, mode=mode, source=source, destination=destination,
               rows=rows, columns=columns, element_bytes=element_bytes,
               source_stride_bytes=source_stride, destination_stride_bytes=destination_stride,
               useful_read_bytes=rows*columns*element_bytes,
               useful_write_bytes=rows*columns*element_bytes, **extra)

    def write(regmap, field, value, address=False):
        if address:
            return [dict(offset=regmap[field + "_DATA_0"], argument=field, part="low32", value=value),
                    dict(offset=regmap[field + "_DATA_1"], argument=field, part="high32", value=value)]
        return [dict(offset=regmap[field + "_DATA_0"], argument=field, value=value)]

    def service(engine, name, arguments, **fields):
        regmap = integer_registers if engine == "integer" else fp_registers
        pointer_fields = ("ACTIVATIONS", "WEIGHTS", "OUTPUT_R") if engine == "integer" else ("X", "Y", "Z", "ACCUMULATORS", "OUTPUT_R", "CODES")
        writes = []
        for field, value in arguments.items():
            writes.extend(write(regmap, field, value, field in pointer_fields))
        writes.append(dict(offset=regmap["AP_CTRL"], argument="AP_START", value=1))
        expected = 11 if engine == "integer" else 19
        if len(writes) != expected:
            raise ValueError("Full generated-register programming count differs.")
        append("service", name, engine=engine, arguments=arguments, register_writes=writes,
               completion=dict(wait="ap_done interrupt; IER=1 only", reads=[
                   dict(offset=regmap["AP_CTRL"], require_mask=2, require_value=2, effect="clear done"),
                   dict(offset=regmap["STATUS_DATA_0"], require_value=0)],
                   clear_interrupt=dict(offset=regmap["ISR"], value=1, effect="toggle done ISR bit once"),
                   any_bus_or_service_error="halt and return unusable status"), **fields)

    allocate("dummy_zero", 4096, "u8", [4096], "pre-zeroed unused-pointer and zero-mask target")
    allocate("token_ids", L*4, "u32", [L], "host window input, including padded positions")
    allocate("type_ids", L*4, "u32", [L], "host window input")
    allocate("attention_mask", L, "u8", [L], "host binary key mask")
    allocate("word_table", model["vocab_size"]*D*4, "f32", [30522,D], "frozen word embedding")
    allocate("type_table", 2*D*4, "f32", [2,D], "frozen type embedding")
    allocate("position_table", 512*D*4, "f32", [512,D], "frozen position embedding; positions0..255 used")
    allocate("input_scales", 24*4, "f32", [24], "frozen per-matrix A8 input scales, copied once per window")
    for layer in range(4):
        for suffix, k, n, group in (("query",D,D,"qkv"),("key",D,D,"qkv"),("value",D,D,"qkv"),
             ("attention_output",D,D,"attention_output"),("ffn_input",D,FF,"ffn_input"),("ffn_output",FF,D,"ffn_output")):
            name = f"layer{layer}_{suffix}"
            group_index = layer*4 + ("qkv","attention_output","ffn_input","ffn_output").index(group)
            allocate(name+"_weights", n*k, "i8_or_packed_i4", [n,k], "fixed maximum-W8 slot; W4 occupies first NK/2 bytes")
            for parameter in ("weight_scale4", "weight_scale8", "bias"):
                allocate(name+"_"+parameter, n*4, "f32", [n], "frozen per-output-channel parameter")
            matrices.append(dict(name=name, inner=k, outputs=n, precision_group=group_index,
                                 input_scale_index=len(matrices)))
    for name in ["embedding_ln"] + [f"layer{i}_{suffix}_ln" for i in range(4) for suffix in ("attention","output")]:
        allocate(name+"_gamma", D*4, "f32", [D], "fixed LayerNorm gamma")
        allocate(name+"_beta", D*4, "f32", [D], "fixed LayerNorm beta")
    for name, n, k in (("pooler",D,D),("classifier",2,D)):
        allocate(name+"_weight", n*k*4, "f32", [n,k], "fixed FP32 output-major weight")
        allocate(name+"_bias", n*4, "f32", [n], "fixed FP32 bias")
    for name in ("embedding_word", "embedding_type", "hidden", "query", "key", "value", "context", "projected", "residual", "normalized"):
        allocate(name, L*D*4, "f32", [L,D], "working buffer; overwritten only after dependencies complete")
    for name in ("ffn_pre_gelu", "ffn_activated"):
        allocate(name, L*FF*4, "f32", [L,FF], "working FFN buffer")
    allocate("a8", L*FF, "i8", [L,FF], "one reusable full integer-input code buffer")
    allocate("accumulator", L*FF*4, "i32", [L,FF], "one reusable full integer-output buffer")
    for name, shape in (("key_head",[L,HD]),("value_head_transposed",[HD,L]),("query_slab",[128,HD]),
                        ("attention_tile",[128,HD]),("scores",[128,L]),("probabilities",[128,L]),("mask_slab",[128,L]),
                        ("pooler_output",[1,D]),("pooled",[1,D]),("logits",[1,2]),("class_probabilities",[1,2])):
        allocate(name, shape[0]*shape[1]*4, "f32", shape, "working or final output")
    dummy = ref("dummy_zero")

    def fp(name, op, rows, width, x=None, y=None, z=None, output=None, codes=None,
           accumulators=None, outputs=0, scale=None, bias=False):
        if rows*width > CAP or (op in ("DOT","SOFTMAX") and width > 512):
            raise ValueError("FP32 service capacity exceeded.")
        if op == "DOT" and (outputs > 64 or outputs*width > CAP or rows*outputs > CAP):
            raise ValueError("FP32 DOT capacity exceeded.")
        args = dict(OP=OPS[op], X=x or dummy, Y=y or dummy, Z=z or dummy,
                    ACCUMULATORS=accumulators or dummy, OUTPUT_R=output or dummy, CODES=codes or dummy,
                    ROWS=rows, WIDTH=width, OUTPUTS=outputs, SCALE=scale or dict(f32_bits=float_bits(1)), DOT_BIAS=int(bias))
        service("fp32", name, args, operation=op)

    def elements(name, op, source, destination, width, y=None, z=None):
        for first in range(0,L,CAP//width):
            rows = min(L-first,CAP//width); size=rows*width*4
            fp(f"{name}_rows{first}",op,rows,width,x=ref(source,first*width*4,size),
               y=ref(y,first*width*4,size) if y and op in ("EMBED_ADD","RESIDUAL_ADD") else ref(y,0,width*4) if y else None,
               z=ref(z,first*width*4,size) if z and op=="EMBED_ADD" else ref(z,0,width*4) if z else None,
               output=ref(destination,first*width*4,size))

    def norm(name, source, destination, parameters):
        elements(name,"LAYER_NORM",source,destination,D,parameters+"_gamma",parameters+"_beta")

    def linear(name, source, destination):
        matrix = next(m for m in matrices if m["name"]==name)
        k,n=matrix["inner"],matrix["outputs"]; group=matrix["precision_group"]
        scale=dict(local_input_scale_index=matrix["input_scale_index"])
        for first in range(0,L,CAP//k):
            rows=min(L-first,CAP//k)
            fp(f"{name}_quantize_rows{first}","QUANTIZE",rows,k,
               x=ref(source,first*k*4,rows*k*4),codes=ref("a8",first*k,rows*k),scale=scale)
        weights=ref(name+"_weights",0,n*k)
        weights["valid_payload_bytes_by_bits"]={"4":n*k//2,"8":n*k}
        args=dict(ACTIVATIONS=ref("a8",0,L*k),WEIGHTS=weights,OUTPUT_R=ref("accumulator",0,L*n*4),
                  ROWS=L,INNER=k,OUTPUTS=n,WEIGHT_BITS=dict(frozen_precision_group=group,allowed=[4,8]))
        service("integer",name,args,precision_group=group)
        scales=dict(select_by_frozen_precision_group=group,
                    alternatives={"4":ref(name+"_weight_scale4",0,n*4),"8":ref(name+"_weight_scale8",0,n*4)})
        for first in range(0,L,CAP//n):
            rows=min(L-first,CAP//n)
            fp(f"{name}_rescale_rows{first}","RESCALE_BIAS",rows,n,y=scales,z=ref(name+"_bias",0,n*4),
               accumulators=ref("accumulator",first*n*4,rows*n*4),output=ref(destination,first*n*4,rows*n*4),scale=scale)

    append("control","validate_window_and_load_inputs", useful_read_bytes=256*(4+4+1)+24*4,
           inputs=[ref("token_ids",0,L*4),ref("type_ids",0,L*4),ref("attention_mask",0,L),ref("input_scales",0,96)],
           checks=["token IDs<30522; type IDs<2; mask bytes0/1; sum(mask)=valid_length in1..256",
                   "24 scales finite and positive; fixed precision map valid; document/window IDs in sequence"],
           local_storage_bytes=256*(4+4+1)+24*4)
    for kind in ("word","type"):
        move("indexed_embedding_rows",f"gather_{kind}_embeddings",ref(kind+"_table",0,buffers[kind+"_table"]["bytes"]),
             ref("embedding_"+kind,0,L*D*4),L,D,source_stride=D*4,destination_stride=D*4,
             index_source="local_token_ids" if kind=="word" else "local_type_ids", index_limit=30522 if kind=="word" else 2)
    append("movement","expand_attention_mask",mode="binary_mask_to_fp32_repeated_rows", rows=128,columns=L,
           source="local_attention_mask",destination=ref("mask_slab",0,128*L*4),
           valid_mask_bits="0x80000000",invalid_mask_bits="0xff7fffff",useful_read_bytes=0,useful_write_bytes=128*L*4)
    elements("embedding_sum","EMBED_ADD","embedding_word","residual",D,"embedding_type","position_table")
    norm("embedding_ln","residual","hidden","embedding_ln")
    for layer in range(4):
        prefix=f"layer{layer}"
        for kind in ("query","key","value"):
            linear(prefix+"_"+kind,"hidden",kind)
        for head in range(H):
            column=head*HD*4
            move("rectangle",f"{prefix}_head{head}_key_gather",ref("key",column,(L-1)*D*4+HD*4),ref("key_head",0,L*HD*4),
                 L,HD,source_stride=D*4,destination_stride=HD*4)
            move("transpose_f32_tiles16",f"{prefix}_head{head}_value_transpose",ref("value",column,(L-1)*D*4+HD*4),
                 ref("value_head_transposed",0,L*HD*4),L,HD,source_stride=D*4,destination_stride=L*4,tile=[16,16])
            for first in (0,128):
                tag=f"{prefix}_head{head}_queries{first}"
                move("rectangle",tag+"_query_gather",ref("query",first*D*4+column,127*D*4+HD*4),ref("query_slab",0,128*HD*4),
                     128,HD,source_stride=D*4,destination_stride=HD*4)
                for panel in range(0,L,64):
                    fp(tag+f"_qk_panel{panel}","DOT",128,HD,x=ref("query_slab",0,128*HD*4),
                       y=ref("key_head",panel*HD*4,64*HD*4),output=ref("attention_tile",0,128*64*4),outputs=64)
                    move("rectangle",tag+f"_scores_scatter{panel}",ref("attention_tile",0,128*64*4),
                         ref("scores",panel*4,127*L*4+64*4),128,64,source_stride=64*4,destination_stride=L*4)
                fp(tag+"_softmax","SOFTMAX",128,L,x=ref("scores",0,128*L*4),y=ref("mask_slab",0,128*L*4),
                   output=ref("probabilities",0,128*L*4),scale=dict(f32_bits=float_bits(8)))
                fp(tag+"_av","DOT",128,L,x=ref("probabilities",0,128*L*4),y=ref("value_head_transposed",0,HD*L*4),
                   output=ref("attention_tile",0,128*HD*4),outputs=HD)
                move("rectangle",tag+"_context_scatter",ref("attention_tile",0,128*HD*4),
                     ref("context",first*D*4+column,127*D*4+HD*4),128,HD,source_stride=HD*4,destination_stride=D*4)
        linear(prefix+"_attention_output","context","projected")
        elements(prefix+"_attention_residual","RESIDUAL_ADD","projected","residual",D,"hidden")
        norm(prefix+"_attention_ln","residual","normalized",prefix+"_attention_ln")
        linear(prefix+"_ffn_input","normalized","ffn_pre_gelu")
        elements(prefix+"_gelu","GELU","ffn_pre_gelu","ffn_activated",FF)
        linear(prefix+"_ffn_output","ffn_activated","projected")
        elements(prefix+"_output_residual","RESIDUAL_ADD","projected","residual",D,"normalized")
        norm(prefix+"_output_ln","residual","hidden",prefix+"_output_ln")
    for output in range(0,D,64):
        fp(f"pooler_panel{output}","DOT",1,D,x=ref("hidden",0,D*4),y=ref("pooler_weight",output*D*4,64*D*4),
           z=ref("pooler_bias",output*4,64*4),output=ref("pooler_output",output*4,64*4),outputs=64,bias=True)
    fp("pooler_tanh","TANH",1,D,x=ref("pooler_output",0,D*4),output=ref("pooled",0,D*4))
    fp("classifier","DOT",1,D,x=ref("pooled",0,D*4),y=ref("classifier_weight",0,2*D*4),
       z=ref("classifier_bias",0,8),output=ref("logits",0,8),outputs=2,bias=True)
    fp("class_softmax","SOFTMAX",1,2,x=ref("logits",0,8),y=ref("dummy_zero",0,8),output=ref("class_probabilities",0,8))
    append("control","read_risk_and_update_document", source=ref("class_probabilities",4,4),useful_read_bytes=4,
           checks=["risk finite, nonnegative and<=1; threshold exact calibrated observed-FP32 u32 bits; host float32 round-trip equality required; zero canonicalized positive"],
           action="window0 initializes document max to +0; subsequent windows require same document/base and contiguous indices; max_risk=max(previous_max,risk); after last window reject iff max_risk>threshold; no early exit",
           comparison="unsigned IEEE-FP32 bit order only after range/sign validation; no approximate FP compare",
           output="sample_id, document_id, window_index, window_risk; on last window document_max_risk, strict_reject, status")
    counts=Counter(c.get("engine") for c in commands if c["kind"]=="service")
    fp_counts=[sum(c.get("operation")==name for c in commands) for name in OPS]
    if counts != {"integer":24,"fp32":411} or fp_counts != EXPECTED_CALLS:
        raise ValueError("Fixed service schedule differs from24 integer+411 FP32 calls.")
    movements=[c for c in commands if c["kind"]=="movement"]
    reads=sum(c.get("useful_read_bytes",0) for c in movements)
    writes=sum(c.get("useful_write_bytes",0) for c in movements)
    service_writes=sum(len(c["register_writes"]) for c in commands if c["kind"]=="service")
    return dict(recorded_utc=datetime.now(timezone.utc).isoformat(),identity="bert_mini_l256_internal_plan_v1",
        status="fixed relative offsets and commands; no physical arena allocation; controller validation is separately recorded",
        arithmetic_boundary="ordered FP32 service panel_cache_batched_attention_v1; signed W4/W8 x A8 integer contract; broader numerical acceptance unresolved",
        arena=dict(planned_bytes=ARENA,used_end_offset=cursor,free_bytes=ARENA-cursor,physical_base=None,allocation_confirmed=False,alignment_bytes=4096),
        address_rule="physical byte address = future allocated arena_base + arena_offset; no physical address assigned",
        precision=dict(groups=[f"layer{i}.{g}" for i in range(4) for g in ("qkv","attention_output","ffn_input","ffn_output")],
                       policy="frozen deployment configuration, not per-request switching; same buffers for every map",
                       packing="output-major contiguous codes; W4 even-k low nibble, odd-k high nibble; signed two-complement codes",
                       fixed_weight_slot_bytes=sum(m["inner"]*m["outputs"] for m in matrices),
                       payload_bytes="1572864+S/2, S=sum(NK) for W8 matrices; reserved slot footprint always3145728",
                       cold_setup="load selected code payload into each fixed slot; load both frozen scale tables; no online repacking"),
        integer_builds=identities,fp32_build=dict(run_record="build/fixed_fp32_hls/run.json",control_source=fp_control.relative_to(ROOT).as_posix(),
             synthesis_recorded_utc=fp_run["recorded_utc"],source_revision="panel_cache_batched_attention_v1"),
        actual_register_offsets=dict(integer=integer_registers,fp32=fp_registers),buffers=buffers,matrices=matrices,
        initial_control_setup=dict(writes_per_engine=[dict(offset=4,value=1),dict(offset=8,value=1)],
                                  precondition="both engines and sequencer reset/idle, IRQ status zero, auto_restart disabled"),
        counts=dict(commands=len(commands),integer_calls=24,fp32_calls=411,fp32_calls_by_op=dict(zip(OPS,fp_counts)),
                    movement_commands=len(movements),control_commands=2,
                    service_argument_start_writes=service_writes,completion_reads=435*2,interrupt_clear_writes=435,
                    interrupt_setup_writes_once_per_reset=4,
                    movement_useful_read_bytes=reads,movement_useful_write_bytes=writes,
                    control_data_read_bytes=256*9+96+4),
        memory_model="None assigned. Byte counts are planned useful traffic; AXI bursts, NoC/DDR stalls, controller cycles and complete latency remain unmeasured.",
        commands=commands)


def markdown(plan):
    c=plan["counts"];a=plan["arena"]
    lines=["# Fixed L256 internal detector command plan","",
        "This is a concrete buffer and ordered-command plan for the existing BERT-Mini detector. The associated sequencer and mover have separate control-only simulation evidence in reports/goal4_internal_integration.md. This plan is not an allocated DDR region, a NoC design, or a measured complete-checking cost.","",
        f"The proposed 64 MiB arena uses {a['used_end_offset']:,} bytes through its last buffer, with {a['free_bytes']:,} bytes spare. Every buffer begins on a 4 KiB boundary. Physical base and allocation remain null. Input IDs, type IDs and mask are supplied by the host; all embedding lookups, detector arithmetic, layout work and final risk/decision control are planned on FPGA.","",
        "The generator reads actual successful integer and FP32 control RTL. All three selected integer variants have identical register maps. The source associations and every byte offset, argument, write and completion action are in `results/goal4/internal_integration/command_plan.json`.","",
        "## Memory and precision","",
        "| Buffer | Relative offset | Bytes | Purpose |","|---|---:|---:|---|"]
    for name,b in plan["buffers"].items():
        lines.append(f"| {name} | 0x{b['offset']:08x} | {b['bytes']} | {b['role']} |")
    lines += ["","Each matrix owns a fixed maximum-W8 slot. W4 occupies its first NK/2 bytes with signed low/high nibbles; W8 occupies NK bytes. Both per-channel weight-scale tables are stored at fixed addresses. A frozen 16-group map selects the weight_bits register and scale pointer. It does not resize or rewire hardware, and is not changed per request. Selected weights/scales are loaded during cold setup, outside warm checking; cold loading and packing remain separately chargeable.","",
        "## Exact execution order","",
        "1. Validate and cache 256 token IDs, 256 type IDs, 256 mask bytes and 24 input scales. Gather 256 word/type embedding rows on PL. Positions 0..255 read the contiguous position table directly. Expand the binary mask into a 128x256 FP32 slab, reused across every head/layer.",
        "2. Add embeddings in (word+type)+position order and apply width 256 LayerNorm. Each encoder matrix uses all 256 rows: quantize input in capacity-bounded chunks, execute one integer call, then rescale/bias in output-width chunks.",
        "3. Per layer/head, gather the 256x64 key panel and transpose the corresponding value head into 64x256. For each 128-query slab, gather queries, run four 64-key QK panels and explicitly scatter their outputs into 128x256 scores. Softmax divides by 8 and applies the repeated mask. One AV dot produces 128x64 output, explicitly scattered into the full context tensor.",
        "4. Attention projection, residual/LN, FFN input, exact-erf GELU, FFN output and residual/LN follow the saved service order. Separate buffers make every dependency explicit; operations remain sequential.",
        "5. Pool the first token with four 64-output DOT panels, tanh, two-output classifier and ordered two-class softmax. Read class 1 probability on PL. Validate finite [0,1] risk and exact calibrated observed-FP32 threshold bits, initialize max to +0 for window 0 and require subsequent contiguous indices with the same document/base, update document max, and compare strictly greater after the last window. Positive IEEE-FP32 bit order is exact after validation and canonicalizing zero. No early reject skips remaining windows; return source IDs/status with the risk.","",
        f"The plan has {c['commands']} commands: 24 integer calls, 411 FP32 calls,{c['movement_commands']} movement operations and 2 input/final-control operations. FP32 op counts are {c['fp32_calls_by_op']}.","",
        "## Generated AXI-Lite contract","",
        "| Argument | Integer offset | FP32 offset |","|---|---|---|"]
    for name in ("AP_CTRL","GIE","IER","ISR","STATUS_DATA_0","STATUS_CTRL"):
        lines.append(f"| {name} | 0x{plan['actual_register_offsets']['integer'][name]:02x} | 0x{plan['actual_register_offsets']['fp32'][name]:02x} |")
    lines += ["","Integer pointer pairs are activations 0x10/14, weights 0x1c/20 and output 0x28/2c; rows 0x34, inner 0x3c, outputs 0x44 and bits 0x4c. FP32 op 0x10; pointer pairs x 0x18/1c, y 0x24/28, z 0x30/34, accumulators 0x3c/40, output 0x48/4c, codes 0x54/58; rows 0x60, width 0x68, outputs 0x70, scale 0x78 and dot_bias 0x80. These offsets are parsed from actual RTL, not invented absolute device addresses.","",
        "The fixed sequencer performs full register programming: 11 writes per integer call and 19 per FP32 call, including start. It waits on the service's done interrupt rather than repeated polling. It reads AP_CTRL to verify/clear done, reads status and requires zero, then toggles ISR bit 0 exactly once. AW and W channels handshake independently and every write waits for B; read waits for R. Bus errors or invalid status stop the command stream with unusable/error status.","",
        f"This gives {c['service_argument_start_writes']} argument/start writes, {c['completion_reads']} completion/status reads and {c['interrupt_clear_writes']} interrupt-clear writes per window; 4 interrupt-enable writes once per reset. The prior provisional ledger's one-read-per-call assumption is replaced by two explicit reads here. No cycles are assigned to these transactions yet.","",
        f"Explicit layout/gather work reads {c['movement_useful_read_bytes']:,} and writes {c['movement_useful_write_bytes']:,} useful bytes per window, plus {c['control_data_read_bytes']:,} control/input bytes read. These bytes exclude service-port traffic already charged by service execution. They are not a DDR-bandwidth estimate.","",
        "## Authorized internal RTL scope","",
        "One fixed-study command sequencer, one 32-bit AXI4-Lite control master selecting the two existing control interfaces, and one bounded 32-bit AXI4 memory mover with 64-bit addresses. The mover supports only this plan's row-copy/strided-rectangle, indexed 256-float embedding-row gather, 16x16 FP32 transpose and binary-mask expansion. Use maximum 16-beat bursts, split at4 KiB boundaries, honor strobes/last/independent backpressure, and allow at most one outstanding transaction per direction. A 16x16 word tile and small index/mask/scale storage are bounded implementation choices, not synthesized resource claims.","",
        "Host preparation must reject a threshold whose float32 round trip differs from the calibrated Python value. Calibration selects an observed FP32 score, so no interpolated or silently rounded threshold is admitted. The supplied bits remain fixed until reset.\n\nThe sequencer latches a deployment model ID, arena base, precision bits and exact threshold until reset; cold setup must load matching weight payloads and scale tables. Per-window sample/document IDs and index are latched on acceptance. The first window initializes max to +0; subsequent indices are contiguous and tied to the same document/base. Final output or any error clears document state, and errors require reset. The sequencer holds these fields and document max; service register values resolve from the arena base and fixed plan. A host-visible start/status mailbox is planned, but its external address and host/Linux transport remain unspecified. The bounded integration simulation connects the sequencer to the actual generated AXI-Lite register modules with bounded completion stubs plus the bounded mover memory model, checking command order/arguments, a mover read error, invalid inputs and final strict-threshold/max behavior. That validates integration control without resynthesizing arithmetic or pretending the stub executes the detector. Reusing the real service RTL for an end-to-end input remains a distinct later integration check.","",
        "No new HLS synthesis, vendor project, NoC configuration, driver, boot image, physical allocation or programming is performed by this generator. A conditional control/layout cycle term is measured separately with explicit arithmetic stubs. Physical DDR/host latency, system resources/timing, actual integrated arithmetic execution and broader numerical acceptance remain unresolved.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true",help="Save the fixed source-derived plan; never executes hardware")
    args=parser.parse_args()
    plan=create_plan(read(ROOT/"configs/project.json"))
    if args.write:
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(plan,indent=2)+"\n")
        REPORT.write_text(markdown(plan))
    print(json.dumps(dict(written=args.write,arena=plan["arena"],counts=plan["counts"]),indent=2))


if __name__=="__main__":
    main()
