CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
TARGET   = mprcu_sim
SRCS     = soft_float_engine.cpp pack_unit.cpp mprcu_top.cpp main.cpp
OBJS     = $(SRCS:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
