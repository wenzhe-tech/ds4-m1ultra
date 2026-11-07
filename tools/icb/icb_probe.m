/* icb_probe.m - does this machine execute indirect compute commands correctly?
 *
 * Written while chasing the decode bottleneck on M1 Ultra / macOS 26, where an
 * indirect command dispatched through an MTLIndirectCommandBuffer produced the
 * right writes but read its input buffers as zero. Nothing in the API surface
 * explained it: encoder type, barriers, inherit modes, storage mode, bind count
 * and binding order were all tried, and a kernel that only writes worked while
 * the same kernel reading a buffer did not.
 *
 * The program is self-checking: it first runs the identical kernel through a
 * normal encoder, so a failure there means the harness or the kernel is wrong
 * rather than ICB. Then it replays the same work through an indirect command
 * buffer, in every configuration that could plausibly matter. A run that ends
 * with "ICB READ: PASS" means the replay layer in ds4_metal.m (DS4_QWEN4_ICB=1)
 * can be re-enabled on this machine.
 *
 * Build and run:
 *     clang -fobjc-arc -fmodules -framework Metal -framework Foundation \
 *         -o icb_probe icb_probe.m && ./icb_probe
 *
 * Exit codes: 0 = ICB reads correctly, 1 = ICB broken, 2 = harness broken.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h>

#define N 256
#define TOL 1e-5f

static const char *kSrc =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    /* Reads buffer 0, writes buffer 1: the case that failed. */
    "kernel void saxpy(device const float *a [[buffer(0)]],\n"
    "                  device float *b [[buffer(1)]],\n"
    "                  uint i [[thread_position_in_grid]]) {\n"
    "    if (i < 256) b[i] = a[i] * 2.0f + 1.0f;\n"
    "}\n"
    /* Arguments delivered through a buffer at a non-zero offset, which is how
     * the replay layer feeds per-command constants. */
    "struct kargs { float m; float pad[63]; };\n"
    "kernel void koff(device const kargs &args [[buffer(0)]],\n"
    "                 device const float *a [[buffer(1)]],\n"
    "                 device float *b [[buffer(2)]],\n"
    "                 uint i [[thread_position_in_grid]]) {\n"
    "    if (i < 256) b[i] = a[i] + args.m;\n"
    "}\n"
    /* Second stage of the chain test: reads what the first stage wrote. */
    "kernel void scale(device const float *b [[buffer(0)]],\n"
    "                  device float *c [[buffer(1)]],\n"
    "                  uint i [[thread_position_in_grid]]) {\n"
    "    if (i < 256) c[i] = b[i] * 10.0f;\n"
    "}\n";

static int g_fail = 0;
static int g_chain_nobarrier_fail = 0;
/* Offsets past 4 GiB are judged separately: they are the reason the decode
 * replay cannot be turned on even though ICB itself works. */
static int g_big_fail = 0;
static int g_big_total = 0;

static void check(const char *label, float got, float want) {
    const int ok = fabsf(got - want) < TOL;
    printf("  %-46s %-22s %s\n", label,
           ok ? "ok" : "MISMATCH",
           ok ? "" : [NSString stringWithFormat:@"got %.4f want %.4f", got, want].UTF8String);
    if (!ok) g_fail++;
}

/* a[i] = i, so the expected output of saxpy is b[i] = 2i + 1. */
static id<MTLBuffer> make_input(id<MTLDevice> dev) {
    id<MTLBuffer> a = [dev newBufferWithLength:N * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    for (int i = 0; i < N; i++) ((float *)a.contents)[i] = (float)i;
    return a;
}

static id<MTLBuffer> make_output(id<MTLDevice> dev) {
    id<MTLBuffer> b = [dev newBufferWithLength:N * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    memset(b.contents, 0, N * sizeof(float));
    return b;
}

static id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib,
                                            const char *name) {
    NSError *e = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { printf("  no function %s\n", name); return nil; }
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = fn;
    pd.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&e];
    if (!pso) printf("  pipeline %s failed: %s\n", name, [[e localizedDescription] UTF8String]);
    return pso;
}

/* Control: the exact same kernel and buffers, but host-encoded. If this fails
 * the problem is not ICB and the rest of the run proves nothing. */
static void test_direct(id<MTLDevice> dev, id<MTLLibrary> lib, id<MTLCommandQueue> q) {
    printf("[harness] direct encode, same kernel and buffers\n");
    id<MTLComputePipelineState> pso = make_pso(dev, lib, "saxpy");
    id<MTLBuffer> a = make_input(dev), b = make_output(dev);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(2, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    check("direct b[0] (want 1)", ((float *)b.contents)[0], 1.0f);
    check("direct b[1] (want 3)", ((float *)b.contents)[1], 3.0f);
    check("direct b[255] (want 511)", ((float *)b.contents)[255], 511.0f);
}

/* One ICB replay configuration. Everything that plausibly matters is a flag
 * here so a run explores the space instead of stopping at the first failure. */
typedef struct {
    const char *name;
    int pipeline_first;    /* set pso before buffers, or after */
    int inherit_buffers;   /* bind on the encoder, not the command */
    int inherit_pipeline;
    int serial;            /* serial instead of concurrent dispatch type */
    int barrier;           /* setBarrier before the dispatch */
    int private_icb;       /* Private instead of Shared storage */
    int bind_count;        /* maxKernelBufferBindCount */
    int use_resource;      /* declare read/write hazards on the encoder */
} probe_cfg;

static void test_icb(id<MTLDevice> dev, id<MTLLibrary> lib, id<MTLCommandQueue> q,
                     probe_cfg cfg) {
    printf("[icb] %s\n", cfg.name);
    id<MTLComputePipelineState> pso = make_pso(dev, lib, "saxpy");
    if (!pso) return;
    id<MTLBuffer> a = make_input(dev), b = make_output(dev);

    MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
    d.commandTypes = cfg.serial ? MTLIndirectCommandTypeConcurrentDispatch
                                : MTLIndirectCommandTypeConcurrentDispatch;
    d.inheritPipelineState = cfg.inherit_pipeline ? YES : NO;
    d.inheritBuffers = cfg.inherit_buffers ? YES : NO;
    d.maxKernelBufferBindCount = (NSUInteger)cfg.bind_count;

    id<MTLIndirectCommandBuffer> icb =
        [dev newIndirectCommandBufferWithDescriptor:d
                                     maxCommandCount:8
                                            options:cfg.private_icb
                                                        ? MTLResourceStorageModePrivate
                                                        : MTLResourceStorageModeShared];
    if (!icb) { printf("  could not create the indirect command buffer\n"); g_fail++; return; }

    id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
    if (cfg.inherit_pipeline) {
        /* The pipeline comes from the executing encoder; setting it per
         * command is a misuse and crashes rather than failing cleanly. */
        if (!cfg.inherit_buffers) {
            [cmd setKernelBuffer:a offset:0 atIndex:0];
            [cmd setKernelBuffer:b offset:0 atIndex:1];
        }
    } else if (cfg.pipeline_first) {
        [cmd setComputePipelineState:pso];
        if (!cfg.inherit_buffers) {
            [cmd setKernelBuffer:a offset:0 atIndex:0];
            [cmd setKernelBuffer:b offset:0 atIndex:1];
        }
    } else {
        if (!cfg.inherit_buffers) {
            [cmd setKernelBuffer:a offset:0 atIndex:0];
            [cmd setKernelBuffer:b offset:0 atIndex:1];
        }
        [cmd setComputePipelineState:pso];
    }
    if (cfg.barrier) [cmd setBarrier];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(2, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc =
        [cb computeCommandEncoderWithDispatchType:cfg.serial ? MTLDispatchTypeSerial
                                                            : MTLDispatchTypeConcurrent];
    if (cfg.use_resource) {
        [enc useResource:icb usage:MTLResourceUsageRead];
        [enc useResource:a usage:MTLResourceUsageRead];
        [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    if (cfg.inherit_buffers) {
        [enc setBuffer:a offset:0 atIndex:0];
        [enc setBuffer:b offset:0 atIndex:1];
    }
    if (cfg.inherit_pipeline) [enc setComputePipelineState:pso];
    [enc executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    check("b[0] (want 1)", ((float *)b.contents)[0], 1.0f);
    check("b[1] (want 3)", ((float *)b.contents)[1], 3.0f);
    check("b[255] (want 511)", ((float *)b.contents)[255], 511.0f);
}


/* Same kernel, arguments read from a buffer at a given offset. The replay
 * layer feeds every command this way, so an offset binding that silently
 * reads the wrong place would explain a wrong result that looks like a
 * broken driver. */
static void test_args_offset(id<MTLDevice> dev, id<MTLLibrary> lib, id<MTLCommandQueue> q,
                             NSUInteger offset) {
    printf("[icb] arguments from a buffer at offset %lu\n", (unsigned long)offset);
    NSError *e = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"koff"];
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = fn;
    pd.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&e];
    if (!pso) { printf("  pipeline failed: %s\n", [[e localizedDescription] UTF8String]); g_fail++; return; }

    id<MTLBuffer> a = make_input(dev), b = make_output(dev);
    /* 256-byte stride, the same as QWEN4_ICB_ARGS_STRIDE. */
    id<MTLBuffer> args = [dev newBufferWithLength:1024 options:MTLResourceStorageModeShared];
    memset(args.contents, 0, 1024);
    *(float *)((char *)args.contents + offset) = 100.0f;

    MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
    d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    d.inheritPipelineState = NO;
    d.inheritBuffers = NO;
    d.maxKernelBufferBindCount = 4;
    id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:d
                                                                  maxCommandCount:8
                                                                         options:MTLResourceStorageModeShared];
    id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
    [cmd setComputePipelineState:pso];
    [cmd setKernelBuffer:args offset:offset atIndex:0];
    [cmd setKernelBuffer:a offset:0 atIndex:1];
    [cmd setKernelBuffer:b offset:0 atIndex:2];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(2, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [enc useResource:icb usage:MTLResourceUsageRead];
    [enc useResource:args usage:MTLResourceUsageRead];
    [enc useResource:a usage:MTLResourceUsageRead];
    [enc useResource:b usage:MTLResourceUsageWrite];
    [enc executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    check("b[1] (want 1 + 100 = 101)", ((float *)b.contents)[1], 101.0f);
    check("b[255] (want 355)", ((float *)b.contents)[255], 355.0f);
}


/* Buffer offsets above 4 GiB. The replay layer binds weights out of one large
 * mapping, so offsets routinely land past the 32-bit boundary; if the command
 * stores the offset in 32 bits it silently reads the wrong place. */
static void test_large_offset(id<MTLDevice> dev, id<MTLLibrary> lib, id<MTLCommandQueue> q,
                              unsigned long long offset) {
    printf("[icb] input buffer bound at offset %llu (%.2f GiB)\n",
           offset, (double)offset / 1073741824.0);
    NSError *e = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"saxpy"];
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = fn;
    pd.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&e];
    if (!pso) { printf("  pipeline failed: %s\n", [[e localizedDescription] UTF8String]); g_fail++; return; }

    /* A pattern at the very start and a different one at the target offset, so
     * a truncated offset reads visibly wrong values rather than zeros. */
    id<MTLBuffer> big = [dev newBufferWithLength:offset + N * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    if (!big) { printf("  could not allocate %llu bytes\n", offset); g_fail++; return; }
    for (int i = 0; i < N; i++) {
        /* Distinguishable markers: a truncated offset lands on the zeros at
         * the start and shows up as 1.0 instead of 201.0. */
        ((float *)big.contents)[i] = 0.0f;
        *(float *)((char *)big.contents + offset + i * sizeof(float)) = 100.0f;
    }
    id<MTLBuffer> b = make_output(dev);

    MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
    d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    d.inheritPipelineState = NO;
    d.inheritBuffers = NO;
    d.maxKernelBufferBindCount = 4;
    id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:d
                                                                  maxCommandCount:8
                                                                         options:MTLResourceStorageModeShared];
    id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
    [cmd setComputePipelineState:pso];
    [cmd setKernelBuffer:big offset:(NSUInteger)offset atIndex:0];
    [cmd setKernelBuffer:b offset:0 atIndex:1];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(2, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [enc useResource:icb usage:MTLResourceUsageRead];
    [enc useResource:big usage:MTLResourceUsageRead];
    [enc useResource:b usage:MTLResourceUsageWrite];
    [enc executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const int ok = fabsf(((float *)b.contents)[1] - 201.0f) < TOL &&
                   fabsf(((float *)b.contents)[255] - 201.0f) < TOL;
    g_big_total++;
    if (!ok) g_big_fail++;
    printf("  read %s (b[1]=%.1f, want 201.0)%s\n", ok ? "correct" : "wrong",
           ((float *)b.contents)[1],
           ok ? "" : "  <-- offset landed elsewhere");
}


/* Same offset test, but the buffer wraps mmap'd memory with no copy, which is
 * how the engine maps the model. A plain allocation passing while this one
 * fails would point at the mapping rather than the offset. */
static void test_large_offset_nocopy(id<MTLDevice> dev, id<MTLLibrary> lib, id<MTLCommandQueue> q,
                                     unsigned long long offset) {
    printf("[icb] mmap-backed buffer bound at offset %llu (%.2f GiB)\n",
           offset, (double)offset / 1073741824.0);
    NSError *e = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"saxpy"];
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = fn;
    pd.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&e];
    if (!pso) { g_fail++; return; }

    const size_t len = (size_t)offset + N * sizeof(float);
    void *mem = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mem == MAP_FAILED) { printf("  mmap failed\n"); g_fail++; return; }
    for (int i = 0; i < N; i++) {
        /* Marker 0 at the start, 100 at the target: a truncated offset that
         * falls back to the start shows up as 1.0 instead of 201.0. */
        ((float *)mem)[i] = 0.0f;
        *(float *)((char *)mem + offset + i * sizeof(float)) = 100.0f;
    }
    id<MTLBuffer> big = [dev newBufferWithBytesNoCopy:mem length:len
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    if (!big) { printf("  no-copy buffer failed\n"); g_fail++; munmap(mem, len); return; }
    id<MTLBuffer> b = make_output(dev);

    MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
    d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    d.inheritPipelineState = NO;
    d.inheritBuffers = NO;
    d.maxKernelBufferBindCount = 4;
    id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:d
                                                                  maxCommandCount:8
                                                                         options:MTLResourceStorageModeShared];
    id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
    [cmd setComputePipelineState:pso];
    [cmd setKernelBuffer:big offset:(NSUInteger)offset atIndex:0];
    [cmd setKernelBuffer:b offset:0 atIndex:1];
    [cmd concurrentDispatchThreadgroups:MTLSizeMake(2, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [enc useResource:icb usage:MTLResourceUsageRead];
    [enc useResource:big usage:MTLResourceUsageRead];
    [enc useResource:b usage:MTLResourceUsageWrite];
    [enc executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const int ok = fabsf(((float *)b.contents)[1] - 201.0f) < TOL &&
                   fabsf(((float *)b.contents)[255] - 201.0f) < TOL;
    g_big_total++;
    if (!ok) g_big_fail++;
    printf("  read %s (b[1]=%.1f, want 201.0)%s\n", ok ? "correct" : "wrong",
           ((float *)b.contents)[1],
           ok ? "" : "  <-- offset landed elsewhere");
}


/* Two dependent commands in one buffer: also proves the replay preserved the
 * order of a chain, which is what a decode step actually needs. */
static void test_chain(id<MTLDevice> dev, id<MTLLibrary> lib, id<MTLCommandQueue> q,
                       int barrier) {
    printf("[icb] two dependent commands (saxpy then scale), second barrier=%s\n",
           barrier ? "yes" : "no");
    id<MTLComputePipelineState> p1 = make_pso(dev, lib, "saxpy");
    id<MTLComputePipelineState> p2 = make_pso(dev, lib, "scale");
    if (!p1 || !p2) return;
    id<MTLBuffer> a = make_input(dev), b = make_output(dev);

    MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
    d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    d.inheritPipelineState = NO;
    d.inheritBuffers = NO;
    d.maxKernelBufferBindCount = 4;
    id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:d
                                                                  maxCommandCount:8
                                                                         options:MTLResourceStorageModeShared];
    id<MTLIndirectComputeCommand> c0 = [icb indirectComputeCommandAtIndex:0];
    [c0 setComputePipelineState:p1];
    [c0 setKernelBuffer:a offset:0 atIndex:0];
    [c0 setKernelBuffer:b offset:0 atIndex:1];
    [c0 concurrentDispatchThreadgroups:MTLSizeMake(2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    id<MTLIndirectComputeCommand> c1 = [icb indirectComputeCommandAtIndex:1];
    /* Commands inside one buffer run concurrently unless a barrier says
     * otherwise. Without this the reader sees the pre-write buffer. */
    if (barrier) [c1 setBarrier];
    [c1 setComputePipelineState:p2];
    [c1 setKernelBuffer:b offset:0 atIndex:0];
    [c1 setKernelBuffer:b offset:0 atIndex:1];   /* in place */
    [c1 concurrentDispatchThreadgroups:MTLSizeMake(2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [enc useResource:icb usage:MTLResourceUsageRead];
    [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [enc executeCommandsInBuffer:icb withRange:NSMakeRange(0, 2)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    check("chained b[1] (want (1*2+1)*10 = 30)", ((float *)b.contents)[1], 30.0f);
    check("chained b[255] (want 5110)", ((float *)b.contents)[255], 5110.0f);
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("no Metal device\n"); return 2; }
        NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
        printf("device      : %s\n", [[dev name] UTF8String]);
        printf("macOS       : %ld.%ld.%ld\n", (long)v.majorVersion, (long)v.minorVersion,
               (long)v.patchVersion);
        printf("has ICB     : %s\n",
               [dev supportsFamily:MTLGPUFamilyApple7] ? "apple7+ (supports family level)" : "unknown");
        printf("\n");

        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                               options:nil
                                                 error:&e];
        if (!lib) {
            printf("library failed: %s\n", [[e localizedDescription] UTF8String]);
            return 2;
        }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        test_direct(dev, lib, q);
        if (g_fail) {
            printf("\nHARNESS BROKEN: the direct path failed, so ICB is not implicated.\n");
            return 2;
        }
        printf("\n");

        probe_cfg cfgs[] = {
            {"basic (pipeline first, concurrent, shared)", 1, 0, 0, 0, 0, 0, 4, 1},
            {"buffers before pipeline",                    0, 0, 0, 0, 0, 0, 4, 1},
            {"barrier before dispatch",                    1, 0, 0, 0, 1, 0, 4, 1},
            {"serial executing encoder",                   1, 0, 0, 1, 0, 0, 4, 1},
            {"inheritBuffers (bind on the encoder)",        1, 1, 0, 0, 0, 0, 4, 1},
            {"inheritPipelineState",                       1, 0, 1, 0, 0, 0, 4, 1},
            {"private indirect command buffer",            1, 0, 0, 0, 0, 1, 4, 1},
            {"maxKernelBufferBindCount = 2",               1, 0, 0, 0, 0, 0, 2, 1},
            {"maxKernelBufferBindCount = 8",               1, 0, 0, 0, 0, 0, 8, 1},
            {"no useResource declaration",                 1, 0, 0, 0, 0, 0, 4, 0},
        };
        const int n_cfg = (int)(sizeof(cfgs) / sizeof(cfgs[0]));
        for (int i = 0; i < n_cfg; i++) {
            const int before = g_fail;
            test_icb(dev, lib, q, cfgs[i]);
            printf("  -> %s\n\n", g_fail == before ? "PASS" : "FAIL");
        }

        /* Without a barrier the second command is lost, which is documented
         * behaviour for concurrent indirect commands rather than a fault: it
         * is counted separately so the verdict stays about ICB itself. */
        const int before_chain = g_fail;
        /* Offsets past 4 GiB are the interesting case: the engine binds
         * weights out of one large mapping, so decode spends its life there. */
        test_large_offset(dev, lib, q, 5ull * 1024 * 1024 * 1024);
        printf("\n");
        test_large_offset_nocopy(dev, lib, q, 2ull * 1024 * 1024 * 1024);
        printf("\n");
        test_large_offset_nocopy(dev, lib, q, 5ull * 1024 * 1024 * 1024);
        printf("\n");

        test_args_offset(dev, lib, q, 0);
        printf("\n");
        test_args_offset(dev, lib, q, 256);
        printf("\n");

        test_chain(dev, lib, q, 0);
        g_chain_nobarrier_fail = g_fail - before_chain;
        g_fail = before_chain;
        printf("\n");
        test_chain(dev, lib, q, 1);
        printf("\n");

        if (g_fail) {
            printf("ICB READ: FAIL (%d checks)\n", g_fail);
            printf("Indirect compute commands do not read their input buffers on this\n"
                   "machine. Keep DS4_QWEN4_ICB unset; the replay layer in ds4_metal.m\n"
                   "stays dormant. Re-run this probe after a macOS or GPU driver update.\n");
            return 1;
        }
        printf("ICB basic commands: PASS\n");
        printf("offsets above 4 GiB: %s (%d of %d offset checks wrong)\n",
               g_big_fail ? "BROKEN" : "ok", g_big_fail, g_big_total);
        printf("\n");
        printf("Single and barrier-ordered indirect commands read correctly here.\n");
        if (g_chain_nobarrier_fail)
            printf("Note: a chain without setBarrier lost %d dependent command(s),\n"
                   "      so every command after the first needs setBarrier.\n",
                   g_chain_nobarrier_fail);
        if (g_big_fail) {
            printf("Even though basic indirect commands work, a binding offset above\n"
                   "4 GiB is truncated to 32 bits: 5 GiB reads the 1 GiB region. The\n"
                   "engine binds its weights from one large mapping, so most decode\n"
                   "commands are affected and DS4_QWEN4_ICB must stay unset until the\n"
                   "weights are reachable through sub-4 GiB bindings.\n");
            return 1;
        }
        printf("The replay layer in ds4_metal.m can be enabled with DS4_QWEN4_ICB=1;\n"
               "validate the output byte-for-byte before\n"
               "trusting the speedup.\n");
    }
    return 0;
}
