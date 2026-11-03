# gguf_project

A study log on model quantization. It starts with one question: **what is actually inside a `.gguf` file?** From there it works down to the bytes, and back up to quantization and how to train through it.

Nothing here is a library or a tool. These are small programs written in the order the questions came up, so the repo is best read as a story.

## The story

### 1. Open the file

I had a quantized model on disk, `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`, and wanted to look inside it. The first attempts used other people's parsers:

- `test.mjs` uses `@huggingface/gguf` to read a remote Llama 2 GGUF over HTTP and print its first tensors.
- `test_read.py` retypes the GGUF value types and field reader in Python, to learn how the format is modelled.

The `test.mjs` output has a surprise in it. The file is named `Q2_K`, but its tensors aren't all stored that way:

| Tensor | `dtype` | Stored as |
|---|---|---|
| `token_embd.weight` | 10 | Q2_K |
| `blk.0.attn_norm.weight` | 0 | F32 (not quantized) |
| `blk.0.ffn_down.weight` | 11 | Q3_K |

So the name on the file describes a mix of formats, not one format used everywhere.

Next I read the header myself. `main.cpp` reads the first 24 bytes straight into a struct:

```
magic "GGUF" | version (u32) | tensor_count (u64) | kv_count (u64)
```

On the Gemma file it prints version 3, 658 tensors and 60 metadata entries. That took about 40 lines of C++ and no libraries.

### 2. Learn what `reinterpret_cast` is doing

That header read depends on `reinterpret_cast<char*>`, so `casting.cpp` is a short detour. It turns a pointer into an integer and back, to show the cast only changes how the same bits are read.

### 3. Build a tiny format

The best way to understand a binary format is to write one. `writer.cpp` produces `toy_model.bin`, a 100-byte model with a header, a name and two float tensors:

| Offset | Size | Contents |
|---|---|---|
| 0 | 12 | Header: `TOY_`, version, tensor count |
| 12 | 32 | Name: `Gemma-Toy-Quant-v1` |
| 44 | 24 | `weight_1`: 6 floats |
| 68 | 32 | `weight_2`: 8 floats |

`reader.cpp` seeks straight to byte 68 and copies out the second tensor, `9.9 8.8 ... 2.2`, without parsing anything before it. A tensor is just bytes at a known offset, and that's why GGUF keeps offsets in its tensor info.

### 4. Quantize

With the file side understood, the next question was how the numbers get smaller.

`quantization.cpp` implements **symmetric** quantization. It splits 256 floats into groups of 32 and gives each group one scale:

```
scale = max(|x|) / 127
q     = round(x / scale)       clamped to [-127, 127]
x'    = q * scale
```

`asym_quants.cpp` implements **asymmetric** (zero-point) quantization. It adds a zero point `z`, so the group's whole min-to-max range maps onto the 256 int8 values:

```
scale = (max - min) / 255
z     = round(-min / scale) - 128
q     = round(x / scale + z)   clamped to [-128, 127]
x'    = (q - z) * scale
```

Both runs use the same test signal, `sin(0.1 * i) * 10`:

| | Bytes per group of 32 | Compression | MSE |
|---|---|---|---|
| Original `float` | 128 | 1x | 0 |
| Symmetric | 36 | 3.6x | 0.000553 |
| Asymmetric | 40 | 3.2x | 0.000170 |

Asymmetric costs 4 more bytes per group (1 for `z`, 3 for struct padding) and gives about 3.3 times lower error here. Each 32-sample group covers about half a sine wave, so most groups are nearly one-sided. Symmetric wastes the half of its range on the side with no values, and asymmetric doesn't.

### 5. Find where it breaks

Asymmetric isn't simply better, at least in this implementation. Replacing the signal with an all-positive one, `5 + 0.01 * (i % 32)`, gives this:

| | MSE |
|---|---|
| Symmetric | 0.000146 |
| Asymmetric | **26.38** |

The formula is fine, but the code isn't. When every value is far from zero, `z` falls well below -128, and storing it in an `int8_t` makes it out of range. Values near 5 come back as about 0.02.

### 6. Train through it

Rounding has a zero gradient almost everywhere, so how can a quantized model be trained? After warming up on PyTorch with `CustomNetwork.py` and `weightvisual.py`, I moved the same math into an autograd function:

- `AsymFakeQuants.py` quantizes and immediately dequantizes the weights on the forward pass ("fake quantization"). On the backward pass it passes the gradient through unchanged, which is the straight-through estimator.
- `withoutquants.py` is the same script with the fake quantization turned off, as a baseline.

Here `z` stays a float, so the int8 overflow from step 5 can't happen.

## Try it

You need `g++`, and optionally Node.js and Python with PyTorch.

```bash
# Build the toy format, then read it back
g++ writer.cpp -o writer && ./writer
g++ reader.cpp -o reader && ./reader

# Compare the two quantization schemes
g++ -fopenmp quantization.cpp -o quantization && ./quantization
g++ -fopenmp asym_quants.cpp -o asym_quants && ./asym_quants

# Read a real GGUF header (edit the hardcoded file_path in main.cpp first)
g++ main.cpp -o main && ./main

# Read tensor info from a remote GGUF
npm install && node test.mjs

# Fake quantization with and without quantizing the weights
pip install torch
python AsymFakeQuants.py
python withoutquants.py
```

`weightvisual.py` also needs `matplotlib` and `seaborn`.

## Files

| Topic | File | What it does |
|---|---|---|
| GGUF | `main.cpp` | Reads a GGUF header by hand |
| | `test.mjs` | Reads remote tensor info with `@huggingface/gguf` |
| | `test_read.py` | Models GGUF value types and fields in Python |
| Binary I/O | `casting.cpp` | Pointer and integer round trip with `reinterpret_cast` |
| | `writer.cpp` | Writes `toy_model.bin` |
| | `reader.cpp` | Reads one tensor by offset |
| Quantization | `quantization.cpp` | Symmetric int8 per group |
| | `asym_quants.cpp` | Asymmetric int8 per group |
| PyTorch | `CustomNetwork.py` | Minimal `nn.Module` |
| | `weightvisual.py` | Heatmap of a layer's weights |
| | `AsymFakeQuants.py` | Fake quantization with a straight-through gradient |
| | `withoutquants.py` | Baseline without quantization |
| Other | `fileparse.py` | CSV parsing practice, not related to quantization |

## Known rough edges

These are left in because this is a study log. They're also the natural next exercises.

- **`asym_quants.cpp`: `z` overflow.** One-sided groups push `z` out of `int8_t` range, as shown in step 5. Storing it in a wider type fixes this, or the range can be stretched to include 0, which keeps `z` in range.
- **`asym_quants.cpp`: wrong zero check.** It tests `max_val == 0` rather than `max_val == min_val`. A constant non-zero group divides by a zero scale, and a group whose maximum is exactly 0 gets flattened to zeros. That early return also leaves `z` unset.
- **`test_read.py` doesn't run yet.** On Python 3.12 it stops with a `NameError`, because `get_type` uses its own class name as an annotation while the class is still being defined. Adding `from __future__ import annotations` should fix it.
- **The fake quantization scripts can't be compared yet.** They use no random seed, so the two printed gradients come from different data. Their target is random too, so the loss doesn't yet measure quantization error, which is what the Korean TODO comment in the scripts asks for.
- **`main.cpp` stops at the header.** The natural next step is to parse the metadata and tensor info that follow it.

## Where to go next

- Implement Q4 and see what happens with only 16 levels per value.
- Work out what the `_K` in `Q4_K_M` adds on top of simple per-group scaling.
- Measure the error on real weights from a GGUF instead of a sine wave.
- Seed the fake quantization scripts and compare how quantized and unquantized training actually differ.
