#include "engine.h"
#include "../internal/binary_io.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace quaml {

using namespace quaml::internal;

Engine::Engine(Feed& feed, Strategy& strategy, Broker& broker)
    : feed(feed), strategy(strategy), broker(broker) {}

void Engine::run() {
    strategy.onStart();
    while (feed.hasNext()) {
        BarView bar = feed.next();
        strategy.onBar(bar);
        equityCurve.push_back(broker.equity(bar.close()));
    }
    strategy.onEnd();
}

void Engine::run_and_save(const std::string& outPath) {
    run();

    std::ofstream f(outPath, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("run_and_save: cannot open file");

    f.write("BQML", 4);

    // Equity curve: bit-cast doubles to int64, then delta+Huffman encode.
    std::vector<int64_t> equityBits(equityCurve.size());
    for (size_t i = 0; i < equityCurve.size(); i++)
        equityBits[i] = doubleToBits(equityCurve[i]);
    writeEncoded64(f, Encode64(equityBits));

    // Trade history: encode timestamps, prices, and qtys separately.
    const auto& trades = broker.history();
    uint32_t n_trades = static_cast<uint32_t>(trades.size());
    writeU32(f, n_trades);

    if (n_trades > 0) {
        std::vector<int64_t> ts(n_trades), prices(n_trades);
        std::vector<int64_t> qtys(n_trades);

        for (uint32_t i = 0; i < n_trades; i++) {
            ts[i]     = trades[i].timestamp;
            prices[i] = doubleToBits(trades[i].price);
            qtys[i]   = trades[i].qty;
        }

        writeEncoded64(f, Encode64(ts));
        writeEncoded64(f, Encode64(prices));
        writeEncoded64(f, Encode64(qtys));
    }
}

void Engine::printStats(double bars_per_year) const {
    if (equityCurve.empty()) {
        std::cout << "No results — run the engine first.\n";
        return;
    }
    std::cout << "--- Backtest Results ---\n";
    std::cout << "Sharpe ratio:  " << stats::sharpe(equityCurve, bars_per_year) << "\n";
    std::cout << "Max drawdown:  " << stats::maxDrawdown(equityCurve) * 100.0 << "%\n";
    std::cout << "CAGR:          " << stats::cagr(equityCurve, bars_per_year) * 100.0 << "%\n";
    std::cout << "Mean return:   " << stats::meanReturn(equityCurve) * 100.0 << "% per bar\n";
    std::cout << "Final equity:  " << equityCurve.back() << "\n";
}

} // namespace quaml
