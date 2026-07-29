#include "broker.h"

namespace quaml {

Broker::Broker(double initial_capital)
    : cash(initial_capital), starting_cash(initial_capital), position(0) {}

void Broker::buy(int qty, double price, int64_t ts) {
    cash -= qty * price;
    position += qty;
    tradeHistory.push_back({ts, price, qty});
}

void Broker::sell(int qty, double price, int64_t ts) {
    cash += qty * price;
    position -= qty;
    tradeHistory.push_back({ts, price, -qty});
}

double Broker::equity(double current_price) const {
    return cash + position * current_price;
}

double Broker::pnl(double current_price) const {
    return equity(current_price) - starting_cash;
}

} // namespace quaml
