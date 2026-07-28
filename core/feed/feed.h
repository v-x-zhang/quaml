#pragma once

#include "../dataframe/dataframe.h"

//we make a feed abstraction so we can have bars or ticks
class Feed{
    const DataFrame& df;
    size_t cursor = 0;

public:
    Feed(const DataFrame& df);
    bool hasNext() const;
    BarView next();
    BarView current() const;

};