#pragma once

#include "../feed/feed.h"
#include "../strategy/strategy.h"
#include <string>

class Engine{
private:

    Feed& feed;
    Strategy& strategy;

public:

    Engine(Feed& feed, Strategy& strategy);
    void run();

    void run_and_save(const std::string&); //writes the data to a bqml file - "backtest qml file"
};