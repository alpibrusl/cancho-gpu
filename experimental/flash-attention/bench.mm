#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
// usage: bench NH NQ T [check]
static float h2f(uint16_t h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }
static uint16_t f2h(float f) { _Float16 x = (_Float16)f; uint16_t h; memcpy(&h, &x, 2); return h; }
int main(int argc, char **argv) {
    @autoreleasepool {
        uint32_t NH = atoi(argv[1]), NQ = atoi(argv[2]), T = atoi(argv[3]); bool check = argc > 4 && argv[4][0] == 'c';
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> qu = [dev newCommandQueue];
        NSString *src = [NSString stringWithContentsOfFile:(getenv("SRC") ? [NSString stringWithUTF8String:getenv("SRC")] : @"flash.metal") encoding:NSUTF8StringEncoding error:nil];
        NSError *err = nil;
        MTLCompileOptions *co = [MTLCompileOptions new]; NSMutableDictionary *pm = [NSMutableDictionary new];
        for (int a = 5; a < argc; ++a) { NSString *d = [NSString stringWithUTF8String:argv[a]]; NSArray *kv = [d componentsSeparatedByString:@"="]; pm[kv[0]] = kv.count > 1 ? kv[1] : @"1"; }
        co.preprocessorMacros = pm;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:co error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
        [fc setConstantValue:&NH type:MTLDataTypeUInt atIndex:0]; [fc setConstantValue:&NQ type:MTLDataTypeUInt atIndex:1]; [fc setConstantValue:&T type:MTLDataTypeUInt atIndex:2];
        id<MTLFunction> fn = [lib newFunctionWithName:@"flash" constantValues:fc error:&err];
        if (!fn) { NSLog(@"%@", err); return 1; }
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) { NSLog(@"%@", err); return 1; }
        size_t ld = (size_t)NH * 128;
        std::vector<uint16_t> q(NQ * ld), k((size_t)T * ld), v((size_t)T * ld);
        srand(3);
        auto rnd = [](float s) { return s * ((rand() / (float)RAND_MAX) * 2 - 1) * 1.7f; };
        for (auto &x : q) x = f2h(rnd(0.3f)); for (auto &x : k) x = f2h(rnd(0.3f)); for (auto &x : v) x = f2h(rnd(1.0f));
        std::vector<float> mask((size_t)NQ * T, 0.0f);
        for (uint32_t i = 0; i < NQ; ++i) for (uint32_t j = 0; j < T; ++j) if (j > i + (T - NQ)) mask[(size_t)i * T + j] = -1e9f;
        id<MTLBuffer> bq = [dev newBufferWithBytes:q.data() length:q.size() * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bk = [dev newBufferWithBytes:k.data() length:k.size() * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bv = [dev newBufferWithBytes:v.data() length:v.size() * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bm = [dev newBufferWithBytes:mask.data() length:mask.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [dev newBufferWithLength:(size_t)NQ * ld * 4 options:MTLResourceStorageModeShared];
        auto run = [&](int reps) {
            id<MTLCommandBuffer> cb = [qu commandBuffer];
            for (int r = 0; r < reps; ++r) {
                id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                [en setComputePipelineState:ps];
                [en setBuffer:bq offset:0 atIndex:0]; [en setBuffer:bk offset:0 atIndex:1]; [en setBuffer:bv offset:0 atIndex:2]; [en setBuffer:bm offset:0 atIndex:3]; [en setBuffer:bo offset:0 atIndex:4];
                [en dispatchThreadgroups:MTLSizeMake(NH, NQ / (getenv("QB") ? atoi(getenv("QB")) : 32), 1) threadsPerThreadgroup:MTLSizeMake((getenv("NSG") ? atoi(getenv("NSG")) : 4) * 32, 1, 1)];
                [en endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) / reps;
        };
        run(1);
        double t = run(10);
        printf("flash attention, %u heads, %u queries, %u keys: %.2f ms a run, %.2f TFLOPS\n", NH, NQ, T, t * 1e3, 4.0 * NH * NQ * (double)T * 128 / t / 1e12);
        if (check) {
            float *o = (float *)bo.contents; double maxerr = 0, rms = 0;
            for (uint32_t h = 0; h < NH; ++h) for (uint32_t i = 0; i < NQ; i += 7) {
                std::vector<double> s(T); double mx = -1e300;
                for (uint32_t j = 0; j < T; ++j) { double a = 0; for (int c = 0; c < 128; ++c) a += (double)h2f(q[i * ld + h * 128 + c]) * h2f(k[j * ld + h * 128 + c]); s[j] = a + mask[(size_t)i * T + j]; mx = fmax(mx, s[j]); }
                double z = 0; for (uint32_t j = 0; j < T; ++j) { s[j] = exp(s[j] - mx); z += s[j]; }
                for (int c = 0; c < 128; ++c) { double a = 0; for (uint32_t j = 0; j < T; ++j) a += s[j] / z * h2f(v[j * ld + h * 128 + c]); maxerr = fmax(maxerr, fabs(a - o[i * ld + h * 128 + c])); rms += a * a; }
            }
            printf("max abs error against float64 attention on every 7th row: %.3g\n", maxerr);
        }
    }
    return 0;
}
