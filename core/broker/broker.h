#pragma once

#include <cstdint>
#include <vector>

//what's a broker for? Well, we need to track the history of our trades.
//We can apply different configs for different broker settings which might include
//things like fees, latency, or other tunable ideas later on like order fill speeds

struct TradeEvent{
    int64_t timestamp;
    double price;
    int qty;
};

class Broker{
    double cash;
    double starting_cash;
    int position; //shares held (negative = short)

    std::vector<TradeEvent> tradeHistory;

public:

    Broker(double initial_capital);
    void buy(int qty, double price);
    void sell(int qty, double price);
    double pnl(double current_price) const;
    double equity(double current_price) const;

};
