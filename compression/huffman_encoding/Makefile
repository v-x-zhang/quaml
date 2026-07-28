CXX = clang++
CXXFLAGS = -std=c++17 -g -Wall -Wextra

TARGET = huffman_main
OBJS = huffman_main.o huffman.o delta_encode.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

huffman_main.o: huffman_main.cpp huffman.h ../delta_encode/delta_encode.h
	$(CXX) $(CXXFLAGS) -c huffman_main.cpp

huffman.o: huffman.cpp huffman.h
	$(CXX) $(CXXFLAGS) -c huffman.cpp

delta_encode.o: ../delta_encode/delta_encode.cpp ../delta_encode/delta_encode.h
	$(CXX) $(CXXFLAGS) -c ../delta_encode/delta_encode.cpp

clean:
	rm -f $(OBJS) $(TARGET)
