#!/usr/bin/env python3
"""Stages the client's shaders for the WebAssembly build.

The renderer loads SPIR-V and hands it to vkCreateShaderModule. In the browser
that call lands in the WebGPU translation layer (src/platform/webgpu), and
WebGPU only takes WGSL - so every shader is translated here, ahead of time, and
the layer finds the translation by hashing the SPIR-V it is given.

For each assets/shaders/<name>.<stage>.glsl this writes to --out:

  <name>.<stage>.spv   what the renderer will load: the committed SPIR-V when
                       there is one, otherwise compiled with glslc exactly as
                       the native build compiles it
  <name>.<stage>.wgsl  the WebGPU translation
  manifest.json        FNV-1a 64 of each .spv -> its .wgsl and the resources it
                       binds, which the layer builds its bind group layouts from

The translation, in order:

  1. The GLSL gets a short header. WebGPU has no point size and no point
     coordinate, so gl_PointSize and gl_PointCoord become ordinary varyings at
     locations 15 and 14 (the client's shaders use 0-7).
  2. glslc compiles it; spirv-opt --split-combined-image-sampler splits each
     sampler2D into a texture and a sampler, because WGSL has no combined type.
  3. The SPIR-V is patched, which naga could not otherwise accept:
       - binding b becomes 2b, and a split-off sampler 2b + 1, so the pair
         the split left on one binding no longer collide
       - the push constant block becomes a uniform buffer at group 0,
         binding PUSH_CONSTANT_BINDING - WebGPU has no push constants, and
         the M2 pipeline already uses all four bind groups WebGPU guarantees
       - NonReadable is dropped from storage buffers; WGSL has no write-only
         storage
  4. naga writes WGSL.

The binding scheme is a contract with the layer: see textureBinding() and
samplerBinding() in src/platform/webgpu/wgpu_layer.hpp. Change one and change
the other.

Needs glslc, spirv-opt and naga (cargo install naga-cli).

    tools/wasm_shaders.py --src assets/shaders --out build-wasm/shaders
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

PUSH_CONSTANT_BINDING = 900
POINT_COORD_LOCATION = 14
POINT_SIZE_LOCATION = 15

STAGES = {"vert": "vertex", "frag": "fragment", "comp": "compute"}

# SPIR-V opcodes, decorations and storage classes used below.
OP_TYPE_IMAGE = 25
OP_TYPE_SAMPLER = 26
OP_TYPE_ARRAY = 28
OP_TYPE_RUNTIME_ARRAY = 29
OP_TYPE_POINTER = 32
OP_VARIABLE = 59
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72
DEC_NON_WRITABLE = 24
DEC_NON_READABLE = 25
DEC_BINDING = 33
DEC_DESCRIPTOR_SET = 34
SC_UNIFORM = 2
SC_PUSH_CONSTANT = 9


def fnv1a64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def point_header(source: str, stage: str) -> str:
    """The GLSL declarations that stand in for the point built-ins."""
    lines = []
    if stage == "vert" and "gl_PointSize" in source:
        lines += [
            f"layout(location = {POINT_SIZE_LOCATION}) out float wowee_pointSize;",
            f"layout(location = {POINT_COORD_LOCATION}) out vec2 wowee_pointCoord;",
            "#define gl_PointSize wowee_pointSize",
        ]
    if stage == "frag" and "gl_PointCoord" in source:
        lines += [
            f"layout(location = {POINT_COORD_LOCATION}) in vec2 wowee_pointCoord;",
            "#define gl_PointCoord wowee_pointCoord",
        ]
    if stage == "frag" and "textureQueryLod" in source:
        # WGSL cannot ask which mip a sample would use, so work it out the way
        # the hardware does: from how fast the texel coordinate changes across
        # the pixel. .x is clamped to the levels that exist, .y is not.
        lines += [
            "vec2 wowee_textureQueryLod(sampler2D s, vec2 uv) {",
            "    vec2 texels = uv * vec2(textureSize(s, 0));",
            "    vec2 dx = dFdx(texels), dy = dFdy(texels);",
            "    float lod = 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-8));",
            "    float top = float(textureQueryLevels(s) - 1);",
            "    return vec2(clamp(lod, 0.0, top), lod);",
            "}",
            "#define textureQueryLod wowee_textureQueryLod",
        ]
    return "\n".join(lines)


def prepare_glsl(source: str, stage: str) -> str:
    header = point_header(source, stage)
    if not header:
        return source
    # After #version and every #extension, which must precede declarations.
    lines = source.split("\n")
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#version") or stripped.startswith("#extension"):
            insert_at = i + 1
    lines.insert(insert_at, header)
    return "\n".join(lines)


def patch_spirv(data: bytes) -> bytes:
    """Rebinds resources and moves push constants into group 0 (see module doc)."""
    words = list(struct.unpack(f"<{len(data) // 4}I", data))
    header, body = words[:5], words[5:]

    insts = []
    i = 0
    while i < len(body):
        count = body[i] >> 16
        insts.append(body[i:i + count])
        i += count

    def op(inst):
        return inst[0] & 0xFFFF

    types = {}
    pointers = {}
    variables = {}
    for inst in insts:
        o = op(inst)
        if o in (OP_TYPE_IMAGE, OP_TYPE_SAMPLER):
            types[inst[1]] = o
        elif o in (OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY):
            types[inst[1]] = ("array", inst[2])
        elif o == OP_TYPE_POINTER:
            pointers[inst[1]] = inst[3]
        elif o == OP_VARIABLE:
            variables[inst[2]] = (inst[1], inst[3])

    def is_sampler(var_id):
        t = pointers.get(variables[var_id][0])
        while isinstance(types.get(t), tuple):
            t = types[t][1]
        return types.get(t) == OP_TYPE_SAMPLER

    push_vars = [v for v, (_, sc) in variables.items() if sc == SC_PUSH_CONSTANT]

    out = []
    last_annotation = -1
    for inst in insts:
        o = op(inst)
        if o == OP_DECORATE and inst[2] == DEC_BINDING and inst[1] in variables:
            b = inst[3]
            inst = [inst[0], inst[1], DEC_BINDING, 2 * b + (1 if is_sampler(inst[1]) else 0)]
        elif o == OP_DECORATE and inst[2] == DEC_NON_READABLE:
            continue
        elif o == OP_MEMBER_DECORATE and inst[3] == DEC_NON_READABLE:
            continue
        elif o == OP_TYPE_POINTER and inst[2] == SC_PUSH_CONSTANT:
            inst = [inst[0], inst[1], SC_UNIFORM, inst[3]]
        elif o == OP_VARIABLE and inst[3] == SC_PUSH_CONSTANT:
            inst = [inst[0], inst[1], inst[2], SC_UNIFORM] + inst[4:]
        out.append(inst)
        if o in (OP_DECORATE, OP_MEMBER_DECORATE):
            last_annotation = len(out) - 1

    if push_vars:
        added = []
        for v in push_vars:
            added.append([(4 << 16) | OP_DECORATE, v, DEC_DESCRIPTOR_SET, 0])
            added.append([(4 << 16) | OP_DECORATE, v, DEC_BINDING, PUSH_CONSTANT_BINDING])
        out[last_annotation + 1:last_annotation + 1] = added

    flat = header + [w for inst in out for w in inst]
    return struct.pack(f"<{len(flat)}I", *flat)


RESOURCE_RE = re.compile(
    r"@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]*)>)?\s+(\w+)\s*:\s*([^;]+);")


def reflect(wgsl: str) -> list:
    """What each @group/@binding in the WGSL is, for the bind group layouts."""
    resources = []
    for m in RESOURCE_RE.finditer(wgsl):
        group, binding = int(m.group(1)), int(m.group(2))
        space, type_ = (m.group(3) or "").replace(" ", ""), m.group(5).strip()
        r = {"group": group, "binding": binding}
        if space == "uniform":
            r["kind"] = "uniform"
        elif space.startswith("storage"):
            r["kind"] = "storage" if "read_write" in space else "read-only-storage"
        elif type_ == "sampler":
            r["kind"] = "sampler"
        elif type_ == "sampler_comparison":
            r["kind"] = "comparison-sampler"
        elif type_.startswith("texture_storage"):
            r["kind"] = "storage-texture"
            r["type"] = type_
        elif type_.startswith("texture"):
            r["kind"] = "texture"
            r["type"] = type_
        else:
            raise ValueError(f"unrecognised resource type: {m.group(0)}")
        resources.append(r)
    return resources


def run(cmd, **kw):
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{result.stdout}{result.stderr}")
    return result


# ImGui's shaders as imgui_impl_vulkan.cpp gives their source, with the one
# change naga needs: the struct varying is two plain ones, at the locations
# the struct's members occupied.
IMGUI_GLSL = {
    "vert": """#version 450 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
layout(push_constant) uniform uPushConstant { vec2 uScale; vec2 uTranslate; } pc;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;
void main() {
    outColor = aColor;
    outUV = aUV;
    gl_Position = vec4(aPos * pc.uScale + pc.uTranslate, 0, 1);
}
""",
    "frag": """#version 450 core
layout(location = 0) out vec4 fColor;
layout(set = 0, binding = 0) uniform sampler2D sTexture;
layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inUV;
void main() {
    fColor = inColor * texture(sTexture, inUV.st);
}
""",
}


IMGUI_ARRAY_RE = re.compile(r"static uint32_t __glsl_shader_(vert|frag)_spv\[\]\s*=\s*\{([^}]*)\}", re.S)


def imgui_shaders(source: Path) -> dict:
    """ImGui's Vulkan backend embeds its two shaders as SPIR-V arrays."""
    found = {}
    for stage, body in IMGUI_ARRAY_RE.findall(source.read_text()):
        words = [int(w, 0) for w in re.findall(r"0x[0-9a-fA-F]+|\d+", body)]
        found[stage] = struct.pack(f"<{len(words)}I", *words)
    return found


def translate(glsl: Path, stage: str, tmp: Path, tools: dict) -> str:
    prepared = tmp / glsl.name
    prepared.write_text(prepare_glsl(glsl.read_text(), stage))
    compiled, split, patched, wgsl = (tmp / f"{glsl.stem}.{n}" for n in
                                      ("spv", "split.spv", "patched.spv", "wgsl"))
    run([tools["glslc"], f"-fshader-stage={STAGES[stage]}", "-O",
         f"-I{glsl.parent}", str(prepared), "-o", str(compiled)])
    run([tools["spirv-opt"], "--split-combined-image-sampler", str(compiled), "-o", str(split)])
    patched.write_bytes(patch_spirv(split.read_bytes()))
    run([tools["naga"], str(patched), str(wgsl)])
    # WGSL rejects a derivative - an implicit-LOD or depth-compare sample, a
    # dFdx - under a branch it cannot prove uniform across the quad. GLSL on
    # Vulkan leaves that to the driver, and the client's shaders do it (the
    # shadow compare inside character.frag's lighting branch, for one); the
    # results are what they are on Vulkan, so the check is turned off.
    return "diagnostic(off, derivative_uniformity);\n" + wgsl.read_text()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--glslc", default="glslc")
    ap.add_argument("--spirv-opt", default="spirv-opt")
    ap.add_argument("--naga", default="naga")
    ap.add_argument("--imgui", type=Path,
                    help="imgui_impl_vulkan.cpp, to translate the shaders it embeds")
    args = ap.parse_args()

    tools = {"glslc": args.glslc, "spirv-opt": args.spirv_opt, "naga": args.naga}
    for name, path in tools.items():
        if not shutil.which(path):
            print(f"error: {name} not found ({path})", file=sys.stderr)
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    failures = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for glsl in sorted(args.src.glob("*.glsl")):
            name = glsl.name[:-len(".glsl")]           # e.g. terrain.frag
            stage = name.rsplit(".", 1)[-1]
            if stage not in STAGES:
                continue
            spv_out = args.out / f"{name}.spv"
            committed = args.src / f"{name}.spv"
            try:
                if committed.exists():
                    shutil.copyfile(committed, spv_out)
                else:
                    run([tools["glslc"], f"-fshader-stage={STAGES[stage]}", "-O",
                         str(glsl), "-o", str(spv_out)])
                wgsl = translate(glsl, stage, tmp, tools)
            except (RuntimeError, ValueError) as e:
                failures.append((name, str(e).strip().splitlines()[-1]))
                continue
            (args.out / f"{name}.wgsl").write_text(wgsl)
            manifest[fnv1a64(spv_out.read_bytes())] = {
                "name": name,
                "stage": STAGES[stage],
                "wgsl": f"{name}.wgsl",
                "resources": reflect(wgsl),
            }

        if args.imgui:
            for stage, spv in imgui_shaders(args.imgui).items():
                name = f"imgui.{stage}"
                glsl = tmp / f"{name}.glsl"
                glsl.write_text(IMGUI_GLSL[stage])
                try:
                    # Found by the hash of ImGui's own SPIR-V, translated from
                    # the equivalent source above.
                    wgsl = translate(glsl, stage, tmp, tools)
                except (RuntimeError, ValueError) as e:
                    failures.append((name, str(e).strip().splitlines()[-1]))
                    continue
                (args.out / f"{name}.wgsl").write_text(wgsl)
                manifest[fnv1a64(spv)] = {
                    "name": name,
                    "stage": STAGES[stage],
                    "wgsl": f"{name}.wgsl",
                    "resources": reflect(wgsl),
                }

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"{len(manifest)} shaders translated into {args.out}")
    for name, why in failures:
        print(f"  FAILED {name}: {why}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
