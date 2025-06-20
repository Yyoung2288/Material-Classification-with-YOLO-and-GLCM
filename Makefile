CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

ifeq ($(shell pkg-config --exists opencv4; echo $$?), 1)
    OPENCV_FLAGS = `pkg-config --cflags --libs opencv`
endif

TARGET = material_classifier
SRCDIR = src
OBJDIR = obj

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

all: $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(OPENCV_FLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_FLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(TARGET)

rebuild: clean all

.PHONY: all clean rebuild 