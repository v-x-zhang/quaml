# Ideas for each core component

## Broker

The core idea of broker is to try to act as the simulated trade facilitation.
We need to track the orders submitted, orders filled, latency profiles, packets dropped, and all that good stuff. 

### Tracking + History

We'll assume anything passed is a valid fill, since fill simulation is an incredibly hard thing to do. I'd like to have varying levels of aggressiveness for what counts as fills. For example, market making can be incredibly difficult to simulate, so we can take the approach that if a fill occured prior or around the same time we would have an order at a better price we would have gotten it instead. Or, maybe we could calculate the average fill price for a specific time. But then, things like VWAP slicing or other tactics make these things a lot more difficult to simulate.

Either way, we'll start by passing the fills to the broker. This will change.

### Order Types

Eventually, I'd like to expand order types to cover the majority of types offered, your LOO LOC, MOO MOCs, etc. Fill or kill, etc.

## Dataframe

I think pandas does some things very well, hence the widespread usage of it in the research jupyter notebook department. My plan is to essentially recreate many parts of it but in C++ and then incorporate them automatically into a backtest. 

## Engine

The engine will be a driver sort of program. It'll facilitate the data loading and timesteps of the backtest, but I think i'll just have it interfacing with the broker to send trades and whatnot based on strategy signals, etc.

## Feed

My issue with backtests based on bars is that for high-frequency trading, it happens to be that packet / tick-based trading or I suppose event-driven trading requires a lot more careful work to be done. I'd like to support both, so this includes maybe supporting the protocol unpacking for binary streams of different exchanges. This means we need to support the packet feeds for ITCH5.0, OUCH5.0, and other exchanges as well. 

The feed abstraction helps us choose between bar-based "rule-trading strategies" and the event-driven based on packets.

## Strategy

The strategy class again helps one define these onBar or onPacket events, and they can test either a bar testing strategy or a tick-based packet one.
