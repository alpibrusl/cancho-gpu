/* canchogpu-metal -- the Metal device path, as cancho can call it.
 *
 * This is an Objective-C++ file that provides a C interface to Metal.
 * It bridges between the C interface that cancho expects and the
 * Objective-C Metal API.
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#import <Metal/Metal.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_MODULES 64
#define MAX_FUNCTIONS 256
#define MAX_BUFFERS 4096
#define MAX_ARGS 64

static struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> libraries[MAX_MODULES];
    int64_t n_libraries;
    id<MTLComputePipelineState> functions[MAX_FUNCTIONS];
    int64_t n_functions;
    id<MTLBuffer> buffers[MAX_BUFFERS];
    int64_t sizes[MAX_BUFFERS];
    int64_t n_buffers;
    id<MTLBuffer> args[MAX_ARGS];
    int64_t arg_offsets[MAX_ARGS];
    int64_t n_args;
    /* The command buffer and encoder that `cgpu_launch_async` fills until `cgpu_sync`. */
    id<MTLCommandBuffer> cmd;
    id<MTLComputeCommandEncoder> enc;
    /* The encoder dispatches concurrently: a launch waits for the ones before it (a
     * buffer barrier) unless `cgpu_overlap` said it does not depend on them. */
    int overlap;
    int fresh;
    /* The last command buffer committed, which `cgpu_sync` waits for. */
    id<MTLCommandBuffer> last;
    char error[4096];
} g;

static int64_t fail(const char *what, const char *detail) {
    snprintf(g.error, sizeof g.error, "%s%s%s", what, detail ? ": " : "", detail ? detail : "");
    return -1;
}

/* A NUL-terminated copy of a cancho byte slice. */
static char *cstr(const char *p, int64_t n) {
    char *s = (char *)malloc((size_t)n + 1);
    if (s) {
        memcpy(s, p, (size_t)n);
        s[n] = 0;
    }
    return s;
}

int64_t cgpu_open(void) {
    if (g.device) {
        return 0;
    }
    
    g.device = MTLCreateSystemDefaultDevice();
    if (!g.device) {
        return fail("no Metal device", "MTLCreateSystemDefaultDevice returned NULL");
    }
    
    g.queue = [g.device newCommandQueue];
    if (!g.queue) {
        return fail("no command queue", "newCommandQueue returned NULL");
    }
    
    return 0;
}

int64_t cgpu_compile(const char *src_p, int64_t src_n) {
    if (!g.device) {
        return fail("cgpu_compile", "the device is not open");
    }
    if (g.n_libraries == MAX_MODULES) {
        return fail("cgpu_compile", "too many modules");
    }
    
    char *src = cstr(src_p, src_n);
    if (!src) {
        return fail("cgpu_compile", "out of memory");
    }
    
    NSString *srcString = [NSString stringWithUTF8String:src];
    free(src);
    
    if (!srcString) {
        return fail("cgpu_compile", "failed to create NSString from source");
    }
    
    NSError *error = nil;
    id<MTLLibrary> library = [g.device newLibraryWithSource:srcString options:nil error:&error];
    
    if (!library) {
        if (error) {
            return fail("the kernel does not compile", [error.localizedDescription UTF8String]);
        }
        return fail("the kernel does not compile", "unknown error");
    }
    
    g.libraries[g.n_libraries] = library;
    return g.n_libraries++;
}

static char prof_name[4096][72];

int64_t cgpu_function(int64_t module, const char *name_p, int64_t name_n) {
    if (module < 0 || module >= g.n_libraries) {
        return fail("cgpu_function", "no such module");
    }
    if (g.n_functions == MAX_FUNCTIONS) {
        return fail("cgpu_function", "too many functions");
    }
    
    char *name = cstr(name_p, name_n);
    if (!name) {
        return fail("cgpu_function", "out of memory");
    }
    
    NSString *funcName = [NSString stringWithUTF8String:name];
    snprintf(prof_name[g.n_functions], sizeof prof_name[0], "%s", name);
    free(name);
    
    if (!funcName) {
        return fail("cgpu_function", "failed to create function name string");
    }
    
    id<MTLFunction> function = [g.libraries[module] newFunctionWithName:funcName];
    if (!function) {
        return fail("cgpu_function", "function not found in library");
    }
    
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [g.device newComputePipelineStateWithFunction:function error:&error];
    
    if (!pipeline) {
        if (error) {
            return fail("cuModuleGetFunction", [error.localizedDescription UTF8String]);
        }
        return fail("cuModuleGetFunction", "unknown error");
    }
    
    g.functions[g.n_functions] = pipeline;
    return g.n_functions++;
}

int64_t cgpu_alloc(int64_t bytes) {
    if (!g.device) {
        return fail("cgpu_alloc", "the device is not open");
    }
    if (g.n_buffers == MAX_BUFFERS || bytes <= 0) {
        return fail("cgpu_alloc", bytes <= 0 ? "an empty buffer" : "too many buffers");
    }
    
    id<MTLBuffer> buffer = [g.device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (!buffer) {
        return fail("cgpu_alloc", "failed to allocate buffer");
    }
    
    g.buffers[g.n_buffers] = buffer;
    g.sizes[g.n_buffers] = bytes;
    return g.n_buffers++;
}

static int64_t buffer_ok(int64_t buf, int64_t n, const char *what) {
    if (buf < 0 || buf >= g.n_buffers) {
        return fail(what, "no such buffer");
    }
    if (n != g.sizes[buf]) {
        return fail(what, "the bytes are not the buffer's size");
    }
    return 0;
}

int64_t cgpu_upload(int64_t buf, const char *p, int64_t n) {
    if (buffer_ok(buf, n, "cgpu_upload")) {
        return -1;
    }
    
    void *bufferContents = [g.buffers[buf] contents];
    if (!bufferContents) {
        return fail("cgpu_upload", "failed to get buffer contents");
    }
    
    memcpy(bufferContents, p, (size_t)n);
    
    // Mark buffer as modified
    [g.buffers[buf] didModifyRange:NSMakeRange(0, (NSUInteger)n)];
    
    return n;
}

// Write `n` bytes into a buffer at `offset`: a model's weights go up a chunk at
// a time, and the activations a token at a time, so a whole-buffer upload is
// not enough.
int64_t cgpu_upload_at(int64_t buf, int64_t offset, const char *p, int64_t n) {
    if (buf < 0 || buf >= g.n_buffers) {
        return fail("cgpu_upload_at", "no such buffer");
    }
    if (offset < 0 || n < 0 || offset > g.sizes[buf] - n) {
        return fail("cgpu_upload_at", "the bytes run past the end of the buffer");
    }
    char *contents = (char *)[g.buffers[buf] contents];
    if (!contents) {
        return fail("cgpu_upload_at", "failed to get buffer contents");
    }
    memcpy(contents + offset, p, (size_t)n);
    return n;
}

/* Fill buffer `buf` with zeros: the states and caches of a model before a new conversation. */
int64_t cgpu_zero(int64_t buf) {
    if (buf < 0 || buf >= g.n_buffers) {
        return fail("cgpu_zero", "no such buffer");
    }
    char *contents = (char *)[g.buffers[buf] contents];
    if (!contents) {
        return fail("cgpu_zero", "failed to get buffer contents");
    }
    memset(contents, 0, (size_t)g.sizes[buf]);
    return 0;
}

int64_t cgpu_download(int64_t buf, char *p, int64_t n) {
    if (buffer_ok(buf, n, "cgpu_download")) {
        return -1;
    }
    
    // For simplicity, we'll use a synchronous approach - read directly from buffer
    // This is a simplified approach - in a real implementation, we'd need to
    // use proper Metal synchronization
    
    // Get the buffer contents directly (this works for CPU-accessible buffers)
    void *bufferContents = [g.buffers[buf] contents];
    if (!bufferContents) {
        return fail("cgpu_download", "failed to get buffer contents");
    }
    
    memcpy(p, bufferContents, (size_t)n);
    
    return n;
}

int64_t cgpu_arg(int64_t buf) {
    if (buf < 0 || buf >= g.n_buffers) {
        return fail("cgpu_arg", "no such buffer");
    }
    if (g.n_args == MAX_ARGS) {
        return fail("cgpu_arg", "too many arguments");
    }
    g.args[g.n_args] = g.buffers[buf];
    g.arg_offsets[g.n_args] = 0;
    return g.n_args++;
}

/* Bind a buffer from a byte offset: a kernel that reads a part of one (the value part
 * of a convolution's output, say) is given just that part as its parameter. */
int64_t cgpu_arg_at(int64_t buf, int64_t offset) {
    if (buf < 0 || buf >= g.n_buffers) {
        return fail("cgpu_arg_at", "no such buffer");
    }
    if (offset < 0 || offset >= g.sizes[buf] || offset % 4 != 0) {
        return fail("cgpu_arg_at", "the offset is outside the buffer or not a multiple of 4");
    }
    if (g.n_args == MAX_ARGS) {
        return fail("cgpu_arg_at", "too many arguments");
    }
    g.args[g.n_args] = g.buffers[buf];
    g.arg_offsets[g.n_args] = offset;
    return g.n_args++;
}

/* Encode a dispatch into the pending command buffer, without waiting: a token's
 * hundreds of kernels then cost one commit instead of one each. Metal orders
 * dispatches that use the same buffers; `cgpu_sync` commits and waits. */
static double prof_ns[4096];
static int64_t prof_n[4096];
int64_t cgpu_prof_dump(void);

int64_t cgpu_launch_async(int64_t fn, int64_t gx, int64_t gy, int64_t threads) {
    if (fn < 0 || fn >= g.n_functions) {
        g.n_args = 0;
        return fail("cgpu_launch_async", "no such function");
    }
    if (!g.cmd) {
        g.cmd = [g.queue commandBuffer];
        if (!g.cmd) {
            g.n_args = 0;
            return fail("cgpu_launch_async", "failed to create command buffer");
        }
        g.enc = [g.cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        g.fresh = 1;
        if (!g.enc) {
            g.cmd = nil;
            g.n_args = 0;
            return fail("cgpu_launch_async", "failed to create compute encoder");
        }
    }
    if (getenv("CANCHOGPU_SKIP") && strstr(prof_name[fn], getenv("CANCHOGPU_SKIP"))) {
        /* Ablation for timing only: leave this kernel out (the outputs are then wrong). */
        g.n_args = 0;
        return 0;
    }
    if (!g.fresh && !g.overlap) {
        [g.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    g.fresh = 0;
    g.overlap = 0;
    [g.enc setComputePipelineState:g.functions[fn]];
    for (int64_t i = 0; i < g.n_args; i++) {
        [g.enc setBuffer:g.args[i] offset:(NSUInteger)g.arg_offsets[i] atIndex:i];
    }
    [g.enc dispatchThreadgroups:MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)threads, 1, 1)];
    g.n_args = 0;
    if (getenv("CANCHOGPU_PROFILE") && getenv("CANCHOGPU_PROFILE")[0] == '2') {
        /* Per-kernel breakdown: run each launch in a buffer of its own and
         * add up its GPU time by function. Slower than the real schedule. */
        [g.enc endEncoding];
        [g.cmd commit];
        [g.cmd waitUntilCompleted];
        prof_ns[fn] += ([g.cmd GPUEndTime] - [g.cmd GPUStartTime]) * 1e9;
        prof_n[fn] += 1;
        g.cmd = nil;
        g.enc = nil;
    }
    return 0;
}

/* Commit what `cgpu_launch_async` encoded, without waiting: the GPU starts on it
 * while the CPU goes on encoding the next. The queue runs command buffers in
 * order, so a later `cgpu_sync` need only wait for the last. */
int64_t cgpu_prof_dump(void) {
    for (int i = 0; i < 4096; i++) {
        if (prof_n[i]) {
            fprintf(stderr, "prof %-44s %4lld launches %8.3f ms\n", prof_name[i], (long long)prof_n[i], prof_ns[i] / 1e6);
            prof_ns[i] = 0;
            prof_n[i] = 0;
        }
    }
    return 0;
}

/* The next `cgpu_launch_async` does not depend on the one before it (nor on any
 * since the last barrier): the GPU may run them together. */
int64_t cgpu_overlap(void) {
    g.overlap = 1;
    return 0;
}

int64_t cgpu_flush(void) {
    if (!g.cmd) {
        return 0;
    }
    [g.enc endEncoding];
    [g.cmd commit];
    g.last = g.cmd;
    g.cmd = nil;
    g.enc = nil;
    return 0;
}

/* Commit what is pending and wait for everything submitted: the GPU time of
 * the last command buffer, in nanoseconds (0 if nothing was submitted). */
int64_t cgpu_sync(void) {
    cgpu_flush();
    if (!g.last) {
        if (getenv("CANCHOGPU_PROFILE") && getenv("CANCHOGPU_PROFILE")[0] == '2') {
            cgpu_prof_dump();
        }
        return 0;
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    [g.last waitUntilCompleted];
    clock_gettime(CLOCK_MONOTONIC, &t1);
    int64_t ns = (int64_t)(([g.last GPUEndTime] - [g.last GPUStartTime]) * 1e9);
    if (getenv("CANCHOGPU_PROFILE")) {
        double wall = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
        fprintf(stderr, "sync: waited %.3f ms, last buffer gpu %.3f ms\n", wall, ns / 1e6);
    }
    g.last = nil;
    if (getenv("CANCHOGPU_PROFILE") && getenv("CANCHOGPU_PROFILE")[0] == '2') {
        cgpu_prof_dump();
    }
    return ns;
}

int64_t cgpu_launch(int64_t fn, int64_t gx, int64_t gy, int64_t threads) {
    if (g.cmd) {
        cgpu_sync();
    }
    if (fn < 0 || fn >= g.n_functions) {
        g.n_args = 0;
        return fail("cgpu_launch", "no such function");
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [g.queue commandBuffer];
    if (!commandBuffer) {
        g.n_args = 0;
        return fail("cgpu_launch", "failed to create command buffer");
    }
    
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (!computeEncoder) {
        [commandBuffer commit];
        g.n_args = 0;
        return fail("cgpu_launch", "failed to create compute encoder");
    }
    
    // Set the compute pipeline state
    [computeEncoder setComputePipelineState:g.functions[fn]];
    
    // Set the arguments
    for (int64_t i = 0; i < g.n_args; i++) {
        [computeEncoder setBuffer:g.args[i] offset:(NSUInteger)g.arg_offsets[i] atIndex:i];
    }
    
    // Dispatch the compute kernel
    MTLSize threadgroupsSize = MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1);
    MTLSize threadsSize = MTLSizeMake((NSUInteger)threads, 1, 1);
    
    [computeEncoder dispatchThreadgroups:threadgroupsSize threadsPerThreadgroup:threadsSize];
    
    // End encoding
    [computeEncoder endEncoding];
    
    // Commit and wait for completion
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    g.n_args = 0;
    return 0;
}

int64_t cgpu_error(char *out, int64_t n) {
    int64_t len = (int64_t)strlen(g.error);
    if (len > n) {
        len = n;
    }
    memcpy(out, g.error, (size_t)len);
    return len;
}

int64_t cgpu_close(void) {
    if (!g.device) {
        return 0;
    }
    
    // Release all resources
    for (int64_t i = 0; i < g.n_buffers; i++) {
        g.buffers[i] = nil;
    }
    for (int64_t i = 0; i < g.n_functions; i++) {
        g.functions[i] = nil;
    }
    for (int64_t i = 0; i < g.n_libraries; i++) {
        g.libraries[i] = nil;
    }
    
    g.queue = nil;
    g.device = nil;
    g.n_buffers = g.n_functions = g.n_libraries = g.n_args = 0;
    
    return 0;
}

#ifdef __cplusplus
}
#endif