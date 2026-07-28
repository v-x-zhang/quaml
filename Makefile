CXX      = clang++
CXXFLAGS = -std=c++17 -g -Wall -Wextra -I.

# --- core objects ---
CORE_OBJS = \
    core/dataframe/dataframe.o \
    core/feed/feed.o \
    core/engine/engine.o

# --- examples ---
EXAMPLES = examples/ma_crossover

all: $(EXAMPLES)

examples/ma_crossover: examples/ma_crossover.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# --- generic compile rules ---
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- dependencies ---
core/dataframe/dataframe.o: core/dataframe/dataframe.cpp core/dataframe/dataframe.h
core/feed/feed.o:           core/feed/feed.cpp           core/feed/feed.h
core/engine/engine.o:       core/engine/engine.cpp       core/engine/engine.h
examples/ma_crossover.o:    examples/ma_crossover.cpp    core/dataframe/dataframe.h \
                            core/feed/feed.h core/strategy/strategy.h core/engine/engine.h

clean:
	rm -f $(CORE_OBJS) examples/ma_crossover.o $(EXAMPLES)
