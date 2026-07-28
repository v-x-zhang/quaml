CXX      = clang++
CXXFLAGS = -std=c++17 -g -Wall -Wextra -I..

TARGET = demo
OBJS   = demo.o encoder.o huffman.o delta_encode.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

demo.o: demo.cpp encoder.h
	$(CXX) $(CXXFLAGS) -c demo.cpp

encoder.o: encoder.cpp encoder.h ../huffman_encoding/huffman.h ../delta_encode/delta_encode.h
	$(CXX) $(CXXFLAGS) -c encoder.cpp

huffman.o: ../huffman_encoding/huffman.cpp ../huffman_encoding/huffman.h
	$(CXX) $(CXXFLAGS) -c ../huffman_encoding/huffman.cpp

delta_encode.o: ../delta_encode/delta_encode.cpp ../delta_encode/delta_encode.h
	$(CXX) $(CXXFLAGS) -c ../delta_encode/delta_encode.cpp

clean:
	rm -f $(OBJS) $(TARGET)
