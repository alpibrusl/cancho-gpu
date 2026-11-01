// The OpenNN half of bench/opennn: the same two-layer sigmoid network as the
// cancho-gpu half, with the same weights, run through OpenNN's own
// `calculate_outputs`.
//
//     forward_opennn DIR R K H REPS
//
// DIR holds x.txt (R*K), w1.txt (H*K, row-major [H,K]), b1.txt (H), w2.txt (H),
// b2.txt (1): plain numbers separated by white space. Writes DIR/opennn_out.txt
// (R numbers) and prints the minimum and the median of REPS timed calls, in
// microseconds. OpenNN keeps a weight matrix as [inputs, outputs], so the
// weights are transposed on the way in; the shapes of its parameter views are
// checked, not assumed.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "opennn/network/network.h"
#include "opennn/network/layers/dense_layer.h"

using namespace opennn;

static std::vector<float> read(const std::string& path, size_t n)
{
    std::ifstream f(path);
    std::vector<float> v;
    float x;
    while (f >> x) v.push_back(x);
    if (v.size() != n) { std::cerr << path << ": " << v.size() << " numbers, expected " << n << "\n"; std::exit(2); }
    return v;
}

static void load_layer(Network& net, int layer, Index in, Index out, const std::vector<float>& w, const std::vector<float>& b)
{
    auto& views = net.get_layer(layer)->get_parameter_views();
    if (views.size() < 2 || views[0].size() != out || views[1].size() != in * out) {
        std::cerr << "layer " << layer << ": unexpected parameter views (" << views.size() << ")\n";
        std::exit(3);
    }
    float* bias = static_cast<float*>(views[0].get_data());
    float* weights = static_cast<float*>(views[1].get_data());
    for (Index o = 0; o < out; ++o) {
        bias[o] = b[o];
        for (Index i = 0; i < in; ++i) weights[i * out + o] = w[o * in + i];
    }
}

int main(int argc, char** argv)
{
    if (argc != 6) { std::cerr << "usage: forward_opennn DIR R K H REPS\n"; return 2; }
    const std::string dir = argv[1];
    const Index R = std::atol(argv[2]), K = std::atol(argv[3]), H = std::atol(argv[4]);
    const int reps = std::atoi(argv[5]);

    const auto x = read(dir + "/x.txt", R * K);
    const auto w1 = read(dir + "/w1.txt", H * K), b1 = read(dir + "/b1.txt", H);
    const auto w2 = read(dir + "/w2.txt", H), b2 = read(dir + "/b2.txt", 1);

    Network net;
    net.add_layer(std::make_unique<opennn::Dense>(Shape{K}, Shape{H}, "Sigmoid", BatchNormalization::No, "d1"));
    net.add_layer(std::make_unique<opennn::Dense>(Shape{H}, Shape{1}, "Sigmoid", BatchNormalization::No, "d2"));
    net.compile();
    load_layer(net, 0, K, H, w1, b1);
    load_layer(net, 1, H, 1, w2, b2);
    net.mark_parameters_changed();

    MatrixR X(R, K);
    for (Index r = 0; r < R; ++r)
        for (Index k = 0; k < K; ++k) X(r, k) = x[r * K + k];

    MatrixR Y = net.calculate_outputs(X);
    if (Y.rows() != R || Y.cols() != 1) { std::cerr << "outputs are " << Y.rows() << "x" << Y.cols() << "\n"; return 4; }
    {
        std::ofstream out(dir + "/opennn_out.txt");
        out.precision(9);
        for (Index r = 0; r < R; ++r) out << Y(r, 0) << (r + 1 < R ? " " : "\n");
    }

    for (int i = 0; i < 20; ++i) net.calculate_outputs(X);  // warm-up, dropped
    std::vector<double> us;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        MatrixR out = net.calculate_outputs(X);
        const auto t1 = std::chrono::steady_clock::now();
        if (out.rows() != R) return 5;
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    std::printf("min_us %.2f median_us %.2f reps %d\n", us.front(), us[us.size() / 2], reps);
    return 0;
}
