# GGUF_project

Small programs for studying model quantization and the GGUF file format.

## Requirements

- `g++` with OpenMP
- Node.js
- Python with PyTorch
- `numpy`, and `matplotlib` and `seaborn`

## Getting Started

```bash
# Write a toy binary model, then read it back
g++ writer.cpp -o writer && ./writer
g++ reader.cpp -o reader && ./reader

# Compare symmetric and asymmetric quantization
g++ -fopenmp quantization.cpp -o quantization && ./quantization
g++ -fopenmp asym_quants.cpp -o asym_quants && ./asym_quants

# Read a GGUF header (set file_path in main.cpp first)
g++ main.cpp -o main && ./main

# Read tensor info from a remote GGUF
npm install && node test.mjs

# Train with and without fake quantization
pip install torch
python AsymFakeQuants.py
python withoutquants.py
```

## Contents

| File | Description |
|---|---|
| `main.cpp` | Reads a GGUF header by hand |
| `test.mjs` | Reads remote tensor info with `@huggingface/gguf` |
| `test_read.py` | Models GGUF value types and fields in Python |
| `casting.cpp` | Pointer and integer round trip with `reinterpret_cast` |
| `writer.cpp` | Writes `toy_model.bin` |
| `reader.cpp` | Reads one tensor from `toy_model.bin` by offset |
| `quantization.cpp` | Symmetric int8 quantization per group |
| `asym_quants.cpp` | Asymmetric int8 quantization per group |
| `CustomNetwork.py` | Minimal `nn.Module` |
| `weightvisual.py` | Heatmap of a layer's weights |
| `AsymFakeQuants.py` | Fake quantization with a straight-through gradient |
| `withoutquants.py` | Baseline without quantization |
| `fileparse.py` | CSV parsing practice, not related to quantization |
