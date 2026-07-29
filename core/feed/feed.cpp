#include "feed.h"

namespace quaml {

Feed::Feed(const DataFrame& df) : df(df) {}

bool Feed::hasNext() const {
    return cursor < df.timestamps.size();
}

BarView Feed::next() {
    return BarView{df, cursor++};
}

BarView Feed::current() const {
    return BarView{df, cursor};
}

} // namespace quaml
