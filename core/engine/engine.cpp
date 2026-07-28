#include "engine.h"

Engine::Engine(Feed& feed, Strategy& strategy) : feed(feed), strategy(strategy){

}

void Engine::run(){
    strategy.onStart();

    while(feed.hasNext()){
        strategy.onBar(feed.next());
    }

    strategy.onEnd();
}