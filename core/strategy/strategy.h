#pragma once

#include "../dataframe/dataframe.h"

namespace quaml {

class Strategy {
public:
    virtual void onStart() {}
    virtual void onEnd()   {}
    virtual void onBar(BarView bar) = 0;
    virtual ~Strategy() = default;
};

} // namespace quaml
