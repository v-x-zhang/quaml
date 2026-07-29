#pragma once

#include "../dataframe/dataframe.h"

namespace quaml {

class Feed {
    const DataFrame& df;
    size_t cursor = 0;

public:
    Feed(const DataFrame& df);
    bool    hasNext()  const;
    BarView next();
    BarView current()  const;
};

} // namespace quaml
