#include "feed.h"

Feed::Feed(const DataFrame& df) : df(df){

}

bool Feed::hasNext() const{
    return cursor < df.timestamps.size();
}

BarView Feed::next(){
    //advance cursor, return current bar
    return BarView{df, cursor++};
}

BarView Feed::current() const{
    return BarView{df, cursor};
}