# 声明编译器和编译选项
NVCC = nvcc
CFLAGS = -std=c++11

# 定义源代码和目标文件的目录
SRCDIR = src
OBJDIR = obj

# 获取所有的.cu源文件
SRCS := $(wildcard $(SRCDIR)/*.cu)

# 将.cu文件的路径替换为.o文件的路径
OBJS := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SRCS))

# 定义目标文件和依赖关系
TARGET = cuda_program

# 默认目标为编译cuda程序
all: $(TARGET)

# 定义目标文件的依赖关系
$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^


# 定义生成目标文件的规则
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c -o $@ $<

$(OBJDIR):
	mkdir -p $(OBJDIR)
# 清除目标文件和编译结果
clean:
	rm -f $(OBJDIR)/*.o $(TARGET)rm -rf $(OBJDIR) $(TARGET)