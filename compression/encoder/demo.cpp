#include <iostream>
#include <random>
#include <vector>
#include "encoder.h"

static std::vector<int32_t> SimulatePrices(int32_t base_cents, int n_ticks, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::discrete_distribution<int32_t> tick_dist({5, 20, 40, 20, 5, 3, 3, 2, 1, 1});
    std::vector<int32_t> moves = {-5, -3, -2, -1, 0, 1, 2, 3, 4, 5};

    std::vector<int32_t> prices;
    prices.reserve(n_ticks);
    int32_t price = base_cents;
    for (int i = 0; i < n_ticks; i++) {
        prices.push_back(price);
        price += moves[tick_dist(rng)];
    }
    return prices;
}

int main() {
    const int32_t BASE_PRICE = 5000000; // $500.00 in centicents
    const int     N_TICKS    = 10000;

    std::vector<int32_t> prices = SimulatePrices(BASE_PRICE, N_TICKS);

    quaml::Encoded enc = quaml::Encode(prices);
    std::vector<int32_t> recovered = quaml::Decode(enc);

    bool ok = (recovered == prices);
    std::cout << "Round-trip OK: " << (ok ? "yes" : "NO - MISMATCH") << "\n\n";

    int raw_bits     = N_TICKS * 32;
    int encoded_bits = 32 + (int)enc.bitstream.size(); // 32 bits for base value
    double ratio     = (double)encoded_bits / raw_bits;

    std::cout << "Ticks:             " << N_TICKS << "\n";
    std::cout << "Raw (32-bit/tick): " << raw_bits     << " bits\n";
    std::cout << "Delta+Huffman:     " << encoded_bits << " bits\n";
    std::cout << "Compression ratio: " << ratio << "x  ("
              << (1.0 - ratio) * 100 << "% savings)\n";

    std::cout << "\nHuffman codebook:\n";
    auto codebook = enc.codebook;
    std::sort(codebook.begin(), codebook.end(),
              [](auto& a, auto& b) { return a.bits.size() < b.bits.size(); });
    for (auto& e : codebook) {
        std::cout << "  delta " << std::showpos << e.symbol
                  << std::noshowpos << " -> " << e.bits << "\n";
    }

    return 0;
}
