CXX      = clang++
CXXFLAGS = -std=c++17 -g -Wall -Wextra -I.

# --- core objects ---
CORE_OBJS = \
    core/dataframe/dataframe.o \
    core/feed/feed.o \
    core/engine/engine.o \
    compression/encoder/encoder.o \
    compression/huffman_encoding/huffman.o \
    compression/delta_encode/delta_encode.o

# --- examples ---
EXAMPLES = examples/ma_crossover

all: $(EXAMPLES)

examples/ma_crossover: examples/ma_crossover.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# --- generic compile rules ---
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- dependencies ---
core/dataframe/dataframe.o: core/dataframe/dataframe.cpp core/dataframe/dataframe.h \
                            compression/encoder/encoder.h
core/feed/feed.o:           core/feed/feed.cpp           core/feed/feed.h
core/engine/engine.o:       core/engine/engine.cpp       core/engine/engine.h
compression/encoder/encoder.o: compression/encoder/encoder.cpp compression/encoder/encoder.h
compression/huffman_encoding/huffman.o: compression/huffman_encoding/huffman.cpp \
                                        compression/huffman_encoding/huffman.h
compression/delta_encode/delta_encode.o: compression/delta_encode/delta_encode.cpp \
                                         compression/delta_encode/delta_encode.h
examples/ma_crossover.o:    examples/ma_crossover.cpp    quaml.h

clean:
	rm -f $(CORE_OBJS) examples/ma_crossover.o $(EXAMPLES)
