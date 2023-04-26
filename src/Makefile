# C++ compiler
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++11

# CUDA compiler
NVCC := nvcc
NVCCFLAGS := -arch=sm_61

# source and object directories
SRCDIR = src
OBJDIR = obj

# Source files
SRCS := $(wildcard $(SRCDIR)/*.cpp $(SRCDIR)/*.cu)
OBJS := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o, $(patsubst $(SRCDIR)%.cu,$(OBJDIR)/%.o,$(SRCS)))

# Target executable
TARGET := my_program

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

