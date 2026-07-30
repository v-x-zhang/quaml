#include <iostream>
#include <fstream>
#include "../quaml.h"

using namespace quaml;

static void generateCSV(const std::string& path, int n_bars) {
    std::ofstream f(path);
    f << "timestamp,open,high,low,close,volume\n";
    double price = 100.0;
    for (int i = 0; i < n_bars; i++) {
        double open  = price;
        double close = price + (i % 7 - 3) * 0.5;
        double high  = std::max(open, close) + 0.2;
        double low   = std::min(open, close) - 0.2;
        f << (1700000000LL + i * 60) << ","
          << open << "," << high << "," << low << "," << close << ","
          << (1000 + i * 10) << "\n";
        price = close;
    }
}

class MACrossover : public Strategy {
    int fast, slow;
    int positions = 0;
    std::vector<double> closes;
    Broker& broker;

    double ma(int period) const {
        double sum = 0;
        for (int i = (int)closes.size() - period; i < (int)closes.size(); i++)
            sum += closes[i];
        return sum / period;
    }

public:
    MACrossover(int fast, int slow, Broker& broker)
        : fast(fast), slow(slow), broker(broker) {}

    void onStart() override {
        std::cout << "MA Crossover started (fast=" << fast << ", slow=" << slow << ")\n";
    }

    void onBar(BarView bar) override {
        closes.push_back(bar.close());
        if ((int)closes.size() < slow) return;

        double fastMA = ma(fast);
        double slowMA = ma(slow);

        if (fastMA > slowMA && positions == 0) {
            broker.buy(1, bar.open(), bar.timestamp());
            positions = 1;
        } else if (fastMA < slowMA && positions == 1) {
            broker.sell(1, bar.open(), bar.timestamp());
            positions = 0;
        }
    }

    void onEnd() override {
        std::cout << "Strategy finished. Final position: "
                  << (positions ? "LONG" : "FLAT") << "\n";
    }
};

int main() {
    const std::string csv = "/tmp/quaml_demo.csv";
    const std::string qml = "/tmp/quaml_demo.qml";
    generateCSV(csv, 100);

    DataFrame df;
    df.readCSV(csv);
    df.saveQML(qml);

    DataFrame df2;
    df2.readQML(qml);

    bool ok = (df.timestamps == df2.timestamps &&
               df.open       == df2.open       &&
               df.close      == df2.close      &&
               df.volume     == df2.volume);
    std::cout << "QML round-trip OK: " << (ok ? "yes" : "NO") << "\n\n";

    Broker      broker(10000.0);
    Feed        feed(df2);
    MACrossover strategy(5, 40, broker);
    Engine      engine(feed, strategy, broker);

    engine.run_and_save("/tmp/backtest.bqml");
    engine.printStats(252.0);

    return 0;
}
