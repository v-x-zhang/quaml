CXX      = clang++
CXXFLAGS = -std=c++17 -g -Wall -Wextra -I.

CORE_OBJS = \
    core/dataframe/dataframe.o \
    core/feed/feed.o \
    core/engine/engine.o \
    core/broker/broker.o \
    core/stats/stats.o \
    ml/features/features.o \
    ml/math/matrix.o \
    compression/encoder/encoder.o \
    compression/huffman_encoding/huffman.o \
    compression/delta_encode/delta_encode.o

EXAMPLES = examples/ma_crossover

all: $(EXAMPLES)

examples/ma_crossover: examples/ma_crossover.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

core/dataframe/dataframe.o: core/dataframe/dataframe.cpp core/dataframe/dataframe.h \
                            compression/encoder/encoder.h
core/feed/feed.o:           core/feed/feed.cpp           core/feed/feed.h
core/engine/engine.o:       core/engine/engine.cpp       core/engine/engine.h
core/broker/broker.o:       core/broker/broker.cpp       core/broker/broker.h
core/stats/stats.o:         core/stats/stats.cpp         core/stats/stats.h
ml/features/features.o:    ml/features/features.cpp     ml/features/features.h
ml/math/matrix.o:          ml/math/matrix.cpp            ml/math/matrix.h
compression/encoder/encoder.o: compression/encoder/encoder.cpp compression/encoder/encoder.h
compression/huffman_encoding/huffman.o: compression/huffman_encoding/huffman.cpp \
                                        compression/huffman_encoding/huffman.h
compression/delta_encode/delta_encode.o: compression/delta_encode/delta_encode.cpp \
                                         compression/delta_encode/delta_encode.h
examples/ma_crossover.o: examples/ma_crossover.cpp quaml.h

clean:
	rm -f $(CORE_OBJS) examples/ma_crossover.o $(EXAMPLES)
