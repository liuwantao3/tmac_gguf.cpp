#include <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <sys/mman.h>

int main() {
    auto dev = MTLCreateSystemDefaultDevice();

    // Test 1: Shared buffer memory location
    id<MTLBuffer> buf = [dev newBufferWithLength:4096 options:MTLStorageModeShared];
    void* gpu_ptr = [buf contents];
    
    // Is it heap? Check if mmap'd or brk'd
    printf("GPU shared buffer address: %p\n", gpu_ptr);
    printf("Is accessible from CPU? %s\n", 
           ({ *(volatile float*)gpu_ptr = 42.0f; *(volatile float*)gpu_ptr == 42.0f ? "yes" : "no"; }) ? "yes" : "no");

    // Test 2: newBufferWithBytes copies data. Let's check addresses
    float data[4] = {1, 2, 3, 4};
    id<MTLBuffer> buf2 = [dev newBufferWithBytes:data length:16 options:MTLStorageModeShared];
    printf("\nnewBufferWithBytes:\n");
    printf("  source data: %p\n", data);
    printf("  buf2 contents: %p\n", [buf2 contents]);
    printf("  Same address? %s\n", data == [buf2 contents] ? "YES" : "NO");
    
    // Test 3: Does CPU see GPU writes instantly?
    printf("\nBefore GPU write: %.1f\n", *(float*)[buf2 contents]);
    // Write via GPU
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    // Use a simple fill kernel
    NSString* src = @"kernel void f(device float* b[[buffer(0)]], uint gid[[thread_position_in_grid]]) { b[gid] = 99.0f; }";
    NSError* e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&e];
    id<MTLFunction> fn = [lib newFunctionWithName:@"f"];
    id<MTLComputePipelineState> pp = [dev newComputePipelineStateWithFunction:fn error:&e];
    [enc setComputePipelineState:pp];
    [enc setBuffer:buf2 offset:0 atIndex:0];
    [enc dispatchThreads:{4,1,1} threadsPerThreadgroup:{4,1,1}];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    
    printf("After GPU write: %.1f (same address, same memory!)\n", *(float*)[buf2 contents]);

    return 0;
}
