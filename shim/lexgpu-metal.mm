/* lexgpu-metal -- the Metal device path, as cancho can call it.
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
#define MAX_BUFFERS 256
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
    int64_t n_args;
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

int64_t lxg_open(void) {
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

int64_t lxg_compile(const char *src_p, int64_t src_n) {
    if (!g.device) {
        return fail("lxg_compile", "the device is not open");
    }
    if (g.n_libraries == MAX_MODULES) {
        return fail("lxg_compile", "too many modules");
    }
    
    char *src = cstr(src_p, src_n);
    if (!src) {
        return fail("lxg_compile", "out of memory");
    }
    
    NSString *srcString = [NSString stringWithUTF8String:src];
    free(src);
    
    if (!srcString) {
        return fail("lxg_compile", "failed to create NSString from source");
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

int64_t lxg_function(int64_t module, const char *name_p, int64_t name_n) {
    if (module < 0 || module >= g.n_libraries) {
        return fail("lxg_function", "no such module");
    }
    if (g.n_functions == MAX_FUNCTIONS) {
        return fail("lxg_function", "too many functions");
    }
    
    char *name = cstr(name_p, name_n);
    if (!name) {
        return fail("lxg_function", "out of memory");
    }
    
    NSString *funcName = [NSString stringWithUTF8String:name];
    free(name);
    
    if (!funcName) {
        return fail("lxg_function", "failed to create function name string");
    }
    
    id<MTLFunction> function = [g.libraries[module] newFunctionWithName:funcName];
    if (!function) {
        return fail("lxg_function", "function not found in library");
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

int64_t lxg_alloc(int64_t bytes) {
    if (!g.device) {
        return fail("lxg_alloc", "the device is not open");
    }
    if (g.n_buffers == MAX_BUFFERS || bytes <= 0) {
        return fail("lxg_alloc", bytes <= 0 ? "an empty buffer" : "too many buffers");
    }
    
    id<MTLBuffer> buffer = [g.device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (!buffer) {
        return fail("lxg_alloc", "failed to allocate buffer");
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

int64_t lxg_upload(int64_t buf, const char *p, int64_t n) {
    if (buffer_ok(buf, n, "lxg_upload")) {
        return -1;
    }
    
    void *bufferContents = [g.buffers[buf] contents];
    if (!bufferContents) {
        return fail("lxg_upload", "failed to get buffer contents");
    }
    
    memcpy(bufferContents, p, (size_t)n);
    
    // Mark buffer as modified
    [g.buffers[buf] didModifyRange:NSMakeRange(0, (NSUInteger)n)];
    
    return n;
}

int64_t lxg_download(int64_t buf, char *p, int64_t n) {
    if (buffer_ok(buf, n, "lxg_download")) {
        return -1;
    }
    
    // For simplicity, we'll use a synchronous approach - read directly from buffer
    // This is a simplified approach - in a real implementation, we'd need to
    // use proper Metal synchronization
    
    // Get the buffer contents directly (this works for CPU-accessible buffers)
    void *bufferContents = [g.buffers[buf] contents];
    if (!bufferContents) {
        return fail("lxg_download", "failed to get buffer contents");
    }
    
    memcpy(p, bufferContents, (size_t)n);
    
    return n;
}

int64_t lxg_arg(int64_t buf) {
    if (buf < 0 || buf >= g.n_buffers) {
        return fail("lxg_arg", "no such buffer");
    }
    if (g.n_args == MAX_ARGS) {
        return fail("lxg_arg", "too many arguments");
    }
    g.args[g.n_args] = g.buffers[buf];
    return g.n_args++;
}

int64_t lxg_launch(int64_t fn, int64_t gx, int64_t gy, int64_t threads) {
    if (fn < 0 || fn >= g.n_functions) {
        g.n_args = 0;
        return fail("lxg_launch", "no such function");
    }
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [g.queue commandBuffer];
    if (!commandBuffer) {
        g.n_args = 0;
        return fail("lxg_launch", "failed to create command buffer");
    }
    
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (!computeEncoder) {
        [commandBuffer commit];
        g.n_args = 0;
        return fail("lxg_launch", "failed to create compute encoder");
    }
    
    // Set the compute pipeline state
    [computeEncoder setComputePipelineState:g.functions[fn]];
    
    // Set the arguments
    for (int64_t i = 0; i < g.n_args; i++) {
        [computeEncoder setBuffer:g.args[i] offset:0 atIndex:i];
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

int64_t lxg_error(char *out, int64_t n) {
    int64_t len = (int64_t)strlen(g.error);
    if (len > n) {
        len = n;
    }
    memcpy(out, g.error, (size_t)len);
    return len;
}

int64_t lxg_close(void) {
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