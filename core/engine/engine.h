#pragma once

#include "../feed/feed.h"
#include "../strategy/strategy.h"

class Engine{
private:

    Feed& feed;
    Strategy& strategy;

public:

    Engine(Feed& feed, Strategy& strategy);
    void run();
};