#include "metal_backend.hpp"
#include <Foundation/Foundation.h>

namespace metal_backend {

inline id<MTLLibrary> load_metal_source(Context& ctx, const char* path) {
    NSString* nsPath = [NSString stringWithUTF8String:path];
    NSError* err = nil;
    NSString* source = [NSString stringWithContentsOfFile:nsPath encoding:NSUTF8StringEncoding error:&err];
    if (!source) {
        printf("[METAL] Read %s: %s\n", path, [[err localizedDescription] UTF8String]);
        return nil;
    }
    id<MTLLibrary> lib = [ctx.device newLibraryWithSource:source options:nil error:&err];
    if (!lib) {
        printf("[METAL] Compile %s: %s\n", path, [[err localizedDescription] UTF8String]);
    }
    return lib;
}

bool init(Context& ctx) {
    ctx.device = MTLCreateSystemDefaultDevice();
    if (!ctx.device) { printf("[METAL] No Metal device\n"); return false; }
    ctx.queue = [ctx.device newCommandQueue];
    if (!ctx.queue) { printf("[METAL] No command queue\n"); return false; }

    id<MTLLibrary> lib = load_metal_source(ctx, "kernels/all_kernels.metal");
    if (!lib) return false;

    auto get_fn = [&](const char* name) -> id<MTLFunction> {
        NSError* err = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
        if (!fn) printf("[METAL] No function: %s\n", name);
        return fn;
    };
    auto make_pipe = [&](id<MTLFunction> fn) -> id<MTLComputePipelineState> {
        if (!fn) return nil;
        NSError* err = nil;
        id<MTLComputePipelineState> p = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        if (err) printf("[METAL] Pipeline: %s\n", [[err localizedDescription] UTF8String]);
        return p;
    };

    ctx.pipe_fp32 = make_pipe(get_fn("mul_mat_fp32"));
    ctx.pipe_q8_0 = make_pipe(get_fn("mul_mat_q8_0"));
    ctx.pipe_q8_0_simd = nil; // DISABLED
    ctx.pipe_q5_0 = make_pipe(get_fn("mul_mat_q5_0"));
    ctx.pipe_q4_k = make_pipe(get_fn("mul_mat_q4_k"));
    ctx.pipe_q6_k = make_pipe(get_fn("mul_mat_q6_k"));
    if (!ctx.pipe_q5_0 || !ctx.pipe_q4_k || !ctx.pipe_q6_k) return false;

    {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        auto make_elem = [&](int op_val) -> id<MTLComputePipelineState> {
            NSError* err = nil;
            [cv setConstantValue:&op_val type:MTLDataTypeInt atIndex:0];
            id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_elem"
                                   constantValues:cv error:&err];
            if (!fn) { printf("[METAL] kernel_elem(%d): %s\n", op_val, [[err localizedDescription] UTF8String]); return nil; }
            return [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        };
        ctx.pipe_elem_add  = make_elem(0);
        ctx.pipe_elem_silu = make_elem(1);
        ctx.pipe_elem_write = make_elem(2);
        if (!ctx.pipe_elem_add || !ctx.pipe_elem_silu || !ctx.pipe_elem_write) return false;
    }
    {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        auto make_qkv = [&](int v) -> id<MTLComputePipelineState> {
            NSError* err = nil;
            [cv setConstantValue:&v type:MTLDataTypeInt atIndex:1];
            id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_fused_qkv"
                                   constantValues:cv error:&err];
            if (!fn) { printf("[METAL] fused_qkv(%d): %s\n", v, [[err description] UTF8String]); return nil; }
            return [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        };
        ctx.pipe_fused_qkv_q5 = make_qkv(0);
        ctx.pipe_fused_qkv_q8 = make_qkv(1);
        if (!ctx.pipe_fused_qkv_q5 || !ctx.pipe_fused_qkv_q8) return false;
    }
    ctx.pipe_fused_ffn_gate_up_q5 = make_pipe(get_fn("kernel_fused_ffn_gate_up_q5"));
    if (!ctx.pipe_fused_ffn_gate_up_q5) printf("[METAL] No fused_ffn_gate_up_q5 (OK)\n");
    ctx.pipe_rope = make_pipe(get_fn("kernel_rope"));
    ctx.pipe_attn = make_pipe(get_fn("kernel_attn"));
    ctx.pipe_flash_attn = nil; // DISABLED
    ctx.pipe_rmsnorm = make_pipe(get_fn("kernel_rmsnorm"));
    if (!ctx.pipe_rope || !ctx.pipe_attn || !ctx.pipe_rmsnorm) return false;

    ctx.initialized = true;
    printf("[METAL] Device: %s\n", [[ctx.device name] UTF8String]);
    printf("[METAL] Max threads/group: %zu\n", ctx.device.maxThreadsPerThreadgroup.width);
    metal_trace_init(ctx);
    return true;
}

void shutdown(Context& ctx) { ctx.initialized = false; }

// ── GPU trace capture ──────────────────────────────────────────
bool capture_start(Context& ctx) {
    MTLCaptureManager* mgr = [MTLCaptureManager sharedCaptureManager];
    if (![mgr supportsDestination:MTLCaptureDestinationDeveloperTools]) return false;
    MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = ctx.device;
    desc.destination = MTLCaptureDestinationDeveloperTools;
    NSError* err;
    return [mgr startCaptureWithDescriptor:desc error:&err];
}

} // namespace metal_backend