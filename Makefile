# 定义编译器和编译选项
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -Wextra -Wpedantic -std=c++11
NVCCFLAGS = -std=c++11

# 源文件目录和目标文件目录
SRCDIR = src
OBJDIR = obj

# 搜寻所有源文件和目标文件
SRCS = $(wildcard $(SRCDIR)/*.cpp $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(filter %.cpp,$(SRCS))) \
       $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(filter %.cu,$(SRCS)))

# 可执行文件和测试文件
EXEC = main
TEST_EXEC = test

# 捕捉所有编译错误
-include $(OBJS:.o=.d)

# 默认编译目标
all: $(EXEC)

# 生成可执行文件
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# 生成目标文件
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -MMD -MP -c $< -o $@

# 编译测试文件
test: test/main.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -Itest -I$(SRCDIR) $^ -o $(TEST_EXEC)

# 清除中间文件和可执行文件
clean:
	rm -rf $(OBJDIR)/*.o $(OBJDIR)/*.d $(EXEC) $(TEST_EXEC)