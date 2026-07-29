#pragma once

#include <cstdint>
#include <vector>

namespace quaml {

struct TradeEvent {
    int64_t timestamp;
    double  price;
    int     qty;        // positive = buy, negative = sell
};

class Broker {
    double cash;
    double starting_cash;
    int    position;    // shares held (negative = short)

    std::vector<TradeEvent> tradeHistory;

public:
    Broker(double initial_capital);

    void   buy (int qty, double price, int64_t ts);
    void   sell(int qty, double price, int64_t ts);

    double pnl   (double current_price) const;
    double equity(double current_price) const;

    const std::vector<TradeEvent>& history() const { return tradeHistory; }
};

} // namespace quaml
