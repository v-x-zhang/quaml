#pragma once

#include "../feed/feed.h"
#include "../strategy/strategy.h"
#include "../broker/broker.h"
#include "../stats/stats.h"
#include <string>
#include <vector>

namespace quaml {

class Engine {
    Feed&     feed;
    Strategy& strategy;
    Broker&   broker;

    std::vector<double> equityCurve;

public:
    Engine(Feed& feed, Strategy& strategy, Broker& broker);

    void run();
    void run_and_save(const std::string& outPath);

    const std::vector<double>& getEquityCurve() const { return equityCurve; }

    void printStats(double bars_per_year = 252.0) const;
};

} // namespace quaml
