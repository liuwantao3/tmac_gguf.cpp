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

    auto get_pipe = [&](const char* name, const char* file) -> id<MTLComputePipelineState> {
        id<MTLLibrary> lib = load_metal_source(ctx, file);
        if (!lib) return nil;
        NSError* err = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
        if (!fn) { printf("[METAL] No function: %s in %s\n", name, file); return nil; }
        return [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    };

    ctx.pipe_fp32 = get_pipe("mul_mat_fp32", "kernels/mul_mat_fp32.metal");
    ctx.pipe_q8_0 = get_pipe("mul_mat_q8_0", "kernels/mul_mat_q8_0.metal");
    ctx.pipe_q8_0_simd = nil; // DISABLED: simdgroup_multiply_accumulate returns zero on M1 Pro
    ctx.pipe_q5_0 = get_pipe("mul_mat_q5_0", "kernels/mul_mat_q5_0.metal");
    ctx.pipe_q4_k = get_pipe("mul_mat_q4_k", "kernels/mul_mat_q4_k.metal");
    ctx.pipe_q6_k = get_pipe("mul_mat_q6_k", "kernels/mul_mat_q6_k.metal");
    if (!ctx.pipe_q5_0 || !ctx.pipe_q4_k || !ctx.pipe_q6_k) return false;

    {
        id<MTLLibrary> lib = load_metal_source(ctx, "kernels/kernel_elem.metal");
        if (!lib) return false;
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        auto make_pipe = [&](int op_val) -> id<MTLComputePipelineState> {
            NSError* err = nil;
            [cv setConstantValue:&op_val type:MTLDataTypeInt atIndex:0];
            id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_elem"
                                   constantValues:cv error:&err];
            if (err) { printf("[METAL] kernel_elem(%d): %s\n", op_val, [[err localizedDescription] UTF8String]); return nil; }
            return [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        };
        ctx.pipe_elem_add  = make_pipe(0);
        ctx.pipe_elem_silu = make_pipe(1);
        ctx.pipe_elem_write = make_pipe(2);
        if (!ctx.pipe_elem_add || !ctx.pipe_elem_silu || !ctx.pipe_elem_write) return false;
    }
    {
        id<MTLLibrary> lib = load_metal_source(ctx, "kernels/kernel_fused_qkv.metal");
        if (!lib) return false;
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        auto make_qkv = [&](int v) -> id<MTLComputePipelineState> {
            NSError* err = nil;
            [cv setConstantValue:&v type:MTLDataTypeInt atIndex:1];
            id<MTLFunction> fn = [lib newFunctionWithName:@"kernel_fused_qkv"
                                   constantValues:cv error:&err];
            if (err) { printf("[METAL] fused_qkv(%d): %s\n", v, [[err description] UTF8String]); return nil; }
            return [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        };
        ctx.pipe_fused_qkv_q5 = make_qkv(0);
        ctx.pipe_fused_qkv_q8 = make_qkv(1);
        if (!ctx.pipe_fused_qkv_q5 || !ctx.pipe_fused_qkv_q8) return false;
    }
    ctx.pipe_fused_ffn_gate_up_q5 = get_pipe("kernel_fused_ffn_gate_up_q5", "kernels/kernel_fused_ffn_gate_up_q5.metal");
    ctx.pipe_rope = get_pipe("kernel_rope", "kernels/kernel_rope.metal");
    ctx.pipe_attn = get_pipe("kernel_attn", "kernels/kernel_attn.metal");
    ctx.pipe_flash_attn = nil; // DISABLED: tmp[8] overflow
    ctx.pipe_rmsnorm = get_pipe("kernel_rmsnorm", "kernels/kernel_rmsnorm.metal");
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