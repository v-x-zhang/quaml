
#include "broker.h"
    

Broker::Broker(double initial_capital) : cash(initial_capital){

}

void Broker::buy(int qty, double price){ // We might need different ordertypes later
    //for now, we say this is an order fill
    cash -= qty * price;
    position += qty;
    tradeHistory.push_back({0, price, qty});
}

void Broker::sell(int qty, double price){
    //for now, we say this is an order fill
    cash += qty * price;
    position -= qty;

    tradeHistory.push_back({0, price, qty});
}

double Broker::pnl(double current_price) const{
    //unrealized pnl
    double current_value = cash + equity(current_price);
    
    return current_value - starting_cash; //current cash + whatever we have * the price we bought it
}

double Broker::equity(double current_price) const{
    return position * current_price;    
}