#include "metal_backend.hpp"
#include <Metal/Metal.h>
#include <iostream>
#include <vector>

// Detailed explanation of Metal GPU programming:
// ========================================
//
// Here's the step-by-step breakdown of running a kernel on GPU:

// Step 1: ALLOCATE BUFFERS ON GPU
// GPU has its own memory. We allocate buffers and copy data from CPU:
// 
//   id<MTLBuffer> bufW = [ctx.device newBufferWithBytes:W.data() length:W.size()*sizeof(float) 
//                                                      options:MTLStorageModeShared];
//                         │
//                         └─ MTLStorageModeShared = CPU and GPU share memory (no copy needed on M1/M2)
//
//   Buffer layout:
//   ┌─────────────┬─────────────┬─────────────┐
//   │ bufW (A)    │ bufX (x)   │ bufY (y)   │  ← GPU memory
//   └─────────────┴─────────────┴─────────────┘
//   These are passed to GPU kernel via [[buffer(0)]], [[buffer(1)]], etc.

// Step 2: CREATE COMMAND BUFFER
// Command buffer = "what to do on GPU"
// We record operations, then submit to GPU:
//
//   id<MTLCommandBuffer> cmdBuf = [ctx.queue commandBuffer];
//   id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
//                        │
//                        └─ Compute encoder = "run a kernel"

// Step 3: SETUP KERNEL AND BIND BUFFERS
// Tell GPU which pipeline (kernel) to run, and which buffers to use:
//
//   [enc setComputePipelineState:ctx.pipeline];  // Which kernel (mul_mat_fp32)
//   [enc setBuffer:bufW offset:0 atIndex:0];  // buffer(0) = A matrix
//   [enc setBuffer:bufX offset:0 atIndex:1];  // buffer(1) = x vector  
//   [enc setBuffer:bufY offset:0 atIndex:2];  // buffer(2) = y output
//   [enc setBuffer:bufParams offset:0 atIndex:3]; // buffer(3) = {rows, cols}
//
//   Mapping to kernel:
//   kernel void mul_mat_fp32(..., uint gid [[thread_position_in_grid]])
//                          │                    │
//   buffer(0) ───────────────┴────────────────────┴─── Each thread knows its ID
//   
//   At kernel: A[[buffer(0)]] means "read from bufW"

// Step 4: LAUNCH THREADS
// How many threads? Grid size:
//
//   [enc dispatchThreads:{rows, 1, 1} threadsPerThreadgroup:{256, 1, 1}];
//
//   ┌─────────────────────────────────────────────────────────────┐
//   Grid = all threads needed                                    │
//   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
//   │ T0 │ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │ ...│T893│T894│T895│
//   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
//   Threadgroup = 256 threads run together on one SIMD unit
//   
//   Thread calculates: y[gid] = row gid of matrix × x vector

// Step 5: EXECUTE
// Submit and wait for completion:
//
//   [cmdBuf commit];        // Send to GPU queue
//   [cmdBuf waitUntilCompleted];  // CPU waits until GPU finishes
//
//   Timeline:
//   ┌──────────┐    ┌───────────────────┐    ┌──────────────────┐
//   │ CPU      │    │ GPU executes       │    │ CPU continues    │
//   │ commits │◀──►│ (parallel)        │───▶│ after wait       │
//   └──────────┘    └───────────────────┘    └──────────────────┘
//   ──────────────────────── async ────────────────────────
//   Non-blocking: CPU can do other work while GPU runs
//
// After kernel:
//   y results are in bufY (same memory as y_cpu due to unified memory)

// Key insight: Unified memory (M1/M2) means no explicit copy!
// Both CPU and GPU access same memory - bufW, bufX, bufY are just "views"
// into the same physical RAM.

void demo_explanation() {
    std::cout << "See code comments above for detailed explanation\n";
    std::cout << "Key points:\n";
    std::cout << "1. Allocate buffers (mtlNewBufferWithBytes)\n";
    std::cout << "2. Create command buffer & encoder\n";
    std::cout << "3. Bind pipeline and buffers\n";
    std::cout << "4. Launch threads (dispatchThreads)\n";
    std::cout << "5. Wait for completion\n";
}

int main() {
    demo_explanation();
    return 0;
}