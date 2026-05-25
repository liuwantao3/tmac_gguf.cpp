#!/usr/bin/env python3
"""
Extract tensors from GGUF and save in simple format for C++ to read.
For Q8_0 tensors, also computes and stores row_max_abs for FPGA precomputation.

TMAC format:
  char magic[4] = "TMAC"
  uint64_t n_tensors (INCLUDES row_max_abs auxiliary tensors)
  for each tensor:
    uint64_t name_len
    char name[name_len]
    uint64_t rows
    uint64_t cols
    uint32_t type (GGUF tensor type)
    uint64_t n_bytes
    uint8_t data[n_bytes]

For Q8_0 tensors, an additional tensor is appended:
    name = original_name + "_row_max_abs"
    rows = original_rows
    cols = 1
    type = 0 (F32)
    data = float32[row_max_abs] per row
"""

import struct
import gguf
from gguf import GGMLQuantizationType
import numpy as np
import os

# Configuration
GGUF_PATH = '/Users/arctic/fpga/models/qwen2-0_5b-instruct-q4_k_m.gguf'
OUTPUT_PATH = '/Users/arctic/fpga/models/model.tmac'

Q8_0 = 8
Q8_BLOCK_SIZE = 32
Q8_BLOCK_BYTES = 34

def fp16_to_float(raw):
    """Convert 16-bit raw integer to float"""
    b0 = raw & 0xFF
    b1 = (raw >> 8) & 0xFF
    sign = (b1 >> 7) & 1
    exp = (b1 & 0x7C) >> 2
    mant = ((b1 & 0x03) << 8) | b0
    if exp == 0:
        return 0.0 if mant == 0 else (1.0 if sign == 0 else -1.0) * (mant / 1024.0) * (2.0 ** -14.0)
    if exp == 31:
        return float('nan') if mant != 0 else (-float('inf') if sign else float('inf'))
    return (1.0 if sign == 0 else -1.0) * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))

def float_to_fp16(val):
    """Convert float to 16-bit raw integer (FP16 bits)"""
    if val == 0.0:
        return 0
    import struct
    f32 = struct.pack('f', val)
    f32_int = struct.unpack('I', f32)[0]
    sign = (f32_int >> 31) & 1
    exp = (f32_int >> 23) & 0xFF
    mant = f32_int & 0x7FFFFF
    if exp == 0:
        return 0
    if exp >= 0x8F:
        exp_f16 = 31
        mant_f16 = 0
    elif exp <= 0x70:
        exp_f16 = 0
        mant_f16 = 0
    else:
        exp_f16 = exp - 127 + 15
        mant_f16 = mant >> 13
    return (sign << 15) | (exp_f16 << 10) | mant_f16

# ===========================================================================
# DEQUANTIZATION (mirrors tmac_gguf.cpp logic)
# ===========================================================================

def dequant_q5_0(data, flat_idx):
    """Dequantize a single value from Q5_0 format. data: uint8 numpy array."""
    block = flat_idx // 32
    offset = flat_idx % 32
    base = block * 22
    d_raw = int(data[base]) | (int(data[base + 1]) << 8)
    d = fp16_to_float(d_raw)
    qh = int(data[base + 2]) | (int(data[base + 3]) << 8) | \
         (int(data[base + 4]) << 16) | (int(data[base + 5]) << 24)
    j = offset if offset < 16 else offset - 16
    qs_byte = int(data[base + 6 + j])
    ql = (qs_byte & 0xF) if offset < 16 else (qs_byte >> 4)
    qh_bit = (qh >> offset) & 1
    q = int((qh_bit << 4) | ql) - 16
    return d * q

def dequant_q6_k(data, flat_idx):
    """Dequantize a single value from Q6_K format. data: uint8 numpy array."""
    QK_K = 256
    BLOCK_BYTES = 210
    block = flat_idx // QK_K
    offset = flat_idx % QK_K
    base = block * BLOCK_BYTES
    d_raw = int(data[base + 208]) | (int(data[base + 209]) << 8)
    d = fp16_to_float(d_raw)
    half = offset // 128
    pos = offset % 128
    l = pos % 32
    sub = pos // 32
    ql_base = base + half * 64
    qh_base = base + 128 + half * 32
    sc_base = base + 192 + half * 8
    if sub == 0:
        ql_byte = int(data[ql_base + l])
        qh_shift = 0
    elif sub == 1:
        ql_byte = int(data[ql_base + l + 32])
        qh_shift = 2
    elif sub == 2:
        ql_byte = int(data[ql_base + l])
        qh_shift = 4
    else:
        ql_byte = int(data[ql_base + l + 32])
        qh_shift = 6
    ql_nibble = (ql_byte & 0xF) if (sub == 0 or sub == 1) else (ql_byte >> 4)
    qh_bits = (int(data[qh_base + l]) >> qh_shift) & 0x3
    q = int((qh_bits << 4) | ql_nibble) - 32
    scale = int(np.int8(data[sc_base + (l // 16) + sub * 2]))
    return d * scale * q

def dequant_q4_k(data, flat_idx):
    """Dequantize a single value from Q4_K format. data: uint8 numpy array."""
    QK_K = 256
    BLOCK_BYTES = 144
    block = flat_idx // QK_K
    offset = flat_idx % QK_K
    base = block * BLOCK_BYTES
    d_raw = int(data[base]) | (int(data[base + 1]) << 8)
    d = fp16_to_float(d_raw)
    dmin_raw = int(data[base + 2]) | (int(data[base + 3]) << 8)
    dmin = fp16_to_float(dmin_raw)
    sub_block = offset // 32
    q_pos = offset % 32
    qs_byte_idx = (sub_block // 2) * 32 + q_pos
    qs_byte = int(data[base + 16 + qs_byte_idx])
    q4 = (qs_byte & 0xF) if (sub_block % 2 == 0) else (qs_byte >> 4)
    scales = data[base + 4 : base + 4 + 12]
    if sub_block < 4:
        sc = int(scales[sub_block]) & 63
        m = int(scales[sub_block + 4]) & 63
    else:
        sc = (int(scales[sub_block + 4]) & 0xF) | ((int(scales[sub_block - 4]) >> 6) << 4)
        m = (int(scales[sub_block + 4]) >> 4) | ((int(scales[sub_block]) >> 6) << 4)
    return d * sc * q4 - dmin * m

# ===========================================================================
# Q8_0 REQUANTIZATION
# ===========================================================================

def quantize_q8_0_block(fp32_block):
    """Quantize 32 FP32 values to Q8_0 format."""
    max_abs = np.max(np.abs(fp32_block))
    if max_abs < 1e-10:
        d = 1.0
    else:
        d = max_abs / 127.0
    q_vals = np.round(fp32_block / d).astype(np.int8)
    d_bits = float_to_fp16(d)
    # Pack: 2 bytes FP16 scale + 32 bytes INT8
    block_bytes = bytearray(34)
    block_bytes[0] = d_bits & 0xFF
    block_bytes[1] = (d_bits >> 8) & 0xFF
    for i in range(32):
        block_bytes[2 + i] = q_vals[i] & 0xFF
    return bytes(block_bytes)

def convert_tensor_to_q8(data, tensor_type, rows, cols):
    """Convert a tensor from any format to Q8_0 format.
    data: raw bytes of the original quantized tensor
    tensor_type: GGUF type (6=Q5_0, 12=Q4_K, 14=Q6_K, 8=Q8_0)
    rows, cols: tensor dimensions

    Returns: (q8_bytes, row_max_abs) where q8_bytes is Q8_0 encoded bytes
             and row_max_abs is a float32 numpy array of shape [rows]
    """
    if tensor_type == Q8_0:
        return data, compute_row_max_abs(np.frombuffer(data, dtype=np.uint8), rows, cols)

    # Select dequant function
    if tensor_type == 6:  # Q5_0
        dequant_fn = dequant_q5_0
    elif tensor_type == 14:  # Q6_K
        dequant_fn = dequant_q6_k
    elif tensor_type == 12:  # Q4_K
        dequant_fn = dequant_q4_k
    else:
        raise ValueError(f"Unsupported tensor type for Q8 conversion: {tensor_type}")

    data_arr = np.frombuffer(data, dtype=np.uint8)
    total_elems = rows * cols
    q8_bytes = bytearray()
    row_max_abs = np.zeros(rows, dtype=np.float32)

    # We process per-32-element Q8_0 block
    # The input is indexed by flat_idx = col + row * cols (GGUF column-major)

    blocks_per_row = (cols + 31) // 32
    out_blocks_per_row = blocks_per_row

    for r in range(rows):
        row_max = 0.0
        for b in range(blocks_per_row):
            # Collect 32 FP32 values for this Q8_0 block
            fp32_block = np.zeros(32, dtype=np.float32)
            for k in range(32):
                col = b * 32 + k
                if col < cols:
                    flat_idx = col + r * cols
                    fp32_block[k] = dequant_fn(data_arr, flat_idx)
                else:
                    fp32_block[k] = 0.0
            # Compute row_max_abs from dequantized values
            block_max = np.max(np.abs(fp32_block))
            if block_max > row_max:
                row_max = block_max
            # Quantize to Q8_0
            q8_bytes.extend(quantize_q8_0_block(fp32_block))
        row_max_abs[r] = row_max

    return bytes(q8_bytes), row_max_abs

def compute_row_max_abs(data, rows, cols):
    """
    Compute row_max_abs for a Q8_0 tensor.
    data: uint8 array of Q8_0 bytes
    rows: number of logical rows (outer dimension in GGML flat indexing)
    cols: number of logical cols (inner dimension)

    Returns: float32 array of shape [rows]
    """
    blocks_per_row = int(cols) // Q8_BLOCK_SIZE
    row_max_abs = np.zeros(rows, dtype=np.float32)

    for r in range(rows):
        row_max = 0.0
        for b in range(blocks_per_row):
            block_base = (r * blocks_per_row + b) * Q8_BLOCK_BYTES
            # Read FP16 scale (2 bytes)
            scale_raw = int(data[block_base]) | (int(data[block_base + 1]) << 8)
            scale = fp16_to_float(scale_raw)
            # Read 32 INT8 values
            vals = data[block_base + 2 : block_base + 2 + 32].astype(np.int32)
            vals[vals >= 128] -= 256  # sign extend
            dequants = vals.astype(np.float32) * scale
            block_max = np.max(np.abs(dequants))
            if block_max > row_max:
                row_max = block_max
        row_max_abs[r] = row_max

    return row_max_abs

def write_tensor(f, name, rows, cols, tensor_type, data):
    """Write a single tensor to the TMAC file"""
    name_bytes = name.encode('utf-8') if isinstance(name, str) else name
    name_len = len(name_bytes)

    f.write(struct.pack('<Q', name_len))
    f.write(name_bytes)
    f.write(struct.pack('<Q', rows))
    f.write(struct.pack('<Q', cols))
    f.write(struct.pack('<I', tensor_type))
    f.write(struct.pack('<Q', len(data)))
    f.write(data)

def main():
    print(f"Reading GGUF: {GGUF_PATH}")
    reader = gguf.GGUFReader(GGUF_PATH)

    print(f'Model: {reader.fields["general.name"].parts[-1]}')
    print(f'Tensors: {len(reader.tensors)}')

    # First pass: identify Q8_0 tensors and compute row_max_abs
    q8_tensors = []
    print("\nIdentifying Q8_0 tensors for row_max_abs computation...")
    for i, t in enumerate(reader.tensors):
        tensor_type = int(t.tensor_type)
        if tensor_type == Q8_0:
            shape = list(t.shape)
            rows = shape[1] if len(shape) > 1 else 1  # GGML: [cols, rows]
            cols = shape[0] if len(shape) > 0 else 1
            name = t.name if isinstance(t.name, bytes) else t.name.encode()
            q8_tensors.append({
                'name': name,
                'rows': rows,
                'cols': cols,
                'data': bytes(t.data),
                'index': i
            })
            print(f"  {len(q8_tensors)}: {name.decode()[:40]} rows={rows} cols={cols}")

    # Compute row_max_abs for all Q8_0 tensors
    print("\nComputing row_max_abs (this takes ~30s per large tensor)...")
    import time
    t0 = time.time()

    for q8 in q8_tensors:
        print(f"  Computing row_max_abs for {q8['name'].decode()[:40]}...")
        t1 = time.time()
        q8['row_max_abs'] = compute_row_max_abs(
            np.frombuffer(q8['data'], dtype=np.uint8),
            q8['rows'],
            q8['cols']
        )
        print(f"    Done in {time.time()-t1:.1f}s, size={q8['row_max_abs'].nbytes} bytes")

    print(f"\nTotal row_max_abs computation: {time.time()-t0:.1f}s")

    # Write TMAC file
    # Total tensors = original + row_max_abs tensors
    n_original = len(reader.tensors)
    n_row_max = len(q8_tensors)
    n_total = n_original + n_row_max

    print(f"\nWriting TMAC: {n_original} original + {n_row_max} row_max_abs = {n_total} total tensors")

    with open(OUTPUT_PATH, 'wb') as f:
        f.write(b'TMAC')
        f.write(struct.pack('<Q', n_total))

        for i, t in enumerate(reader.tensors):
            name = t.name if isinstance(t.name, bytes) else t.name.encode()
            name_len = len(name)

            shape = list(t.shape)
            rows = shape[1] if len(shape) > 1 else 1
            cols = shape[0] if len(shape) > 0 else 1

            tensor_type = int(t.tensor_type)
            raw_data = bytes(t.data)
            n_bytes = len(raw_data)

            write_tensor(f, name, rows, cols, tensor_type, raw_data)

            if i < 5 or tensor_type not in [0, 6]:
                gg = GGMLQuantizationType(t.tensor_type)
                type_name = gg.name if hasattr(gg, 'name') else str(tensor_type)
                print(f'{i}: {name.decode()[:40]}')
                print(f'   type={tensor_type} ({type_name}), bytes={n_bytes}')

        # Write row_max_abs tensors
        for q8 in q8_tensors:
            row_max_name = q8['name'] + b'_row_max_abs'
            row_max_data = q8['row_max_abs'].tobytes()

            write_tensor(f, row_max_name, q8['rows'], 1, 0, row_max_data)
            print(f'  + row_max_abs: {row_max_name.decode()[:40]} bytes={len(row_max_data)}')

    size = os.path.getsize(OUTPUT_PATH)
    print(f'\nDone! File size: {size / (1024*1024):.1f} MB')

if __name__ == '__main__':
    main()