// Time an emitted NVFP4 matvec kernel (any M) over the 17408x5120 shape.
//
//     bwk <kernel.metal> <kernel name> <M> <rows per threadgroup> [N K]
//
// Arguments are the emitter's: x[M,K], wq, ws, wg, y[M,N]. 16 copies of the
// weights are used in turn so that none is found in the system cache.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 5) { fprintf(stderr, "usage: bwk file.metal name M rows [N K]\n"); return 2; }
        int M = atoi(argv[3]), RPT = atoi(argv[4]);
        int N = argc > 5 ? atoi(argv[5]) : 17408, K = argc > 6 ? atoi(argv[6]) : 5120;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&err];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { NSLog(@"%@", err); return 1; }
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:argv[2]]] error:&err];
        if (!ps) { NSLog(@"%@", err); return 1; }
        size_t nq = (size_t)N * K / 2, ns = (size_t)N * K / 16;
        std::vector<unsigned char> wq(nq), ws(ns);
        std::vector<float> x((size_t)M * K), wg(N, 1.0f / 16384.0f);
        srand(1);
        for (auto &b : wq) b = rand();
        for (auto &b : ws) b = 0x30 + (rand() & 0x1f);
        for (auto &v : x) v = (rand() / (float)RAND_MAX) - 0.5f;
        const int COPIES = 16;
        std::vector<id<MTLBuffer>> bq, bs;
        for (int c = 0; c < COPIES; c++) {
            bq.push_back([dev newBufferWithBytes:wq.data() length:nq options:0]);
            bs.push_back([dev newBufferWithBytes:ws.data() length:ns options:0]);
        }
        id<MTLBuffer> bx = [dev newBufferWithBytes:x.data() length:x.size() * 4 options:0];
        id<MTLBuffer> bg = [dev newBufferWithBytes:wg.data() length:N * 4 options:0];
        id<MTLBuffer> by = [dev newBufferWithLength:(size_t)M * N * 4 options:0];
        auto launch = [&](int reps) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:bx offset:0 atIndex:0]; [e setBuffer:bg offset:0 atIndex:3]; [e setBuffer:by offset:0 atIndex:4]; [e setBuffer:by offset:0 atIndex:5];
            for (int i = 0; i < reps; i++) {
                [e setBuffer:bq[i % COPIES] offset:0 atIndex:1]; [e setBuffer:bs[i % COPIES] offset:0 atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(N / RPT, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return [cb GPUEndTime] - [cb GPUStartTime];
        };
        launch(3);
        double t = launch(100) / 100;
        printf("%-34s M=%d  %7.1f us  %6.1f GB/s of weights\n", argv[2], M, t * 1e6, (double)(nq + ns) / t / 1e9);
    }
    return 0;
}
