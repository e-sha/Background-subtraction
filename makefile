program_name := test
source_dirs := src
source_dirs := $(addprefix ../,$(source_dirs))
search_wildcards := $(addsuffix /*.cpp,$(source_dirs))
additional_flags := -lopencv_core -lopencv_highgui -lopencv_imgproc -fopenmp -lstdc++ -lm
$(program_name): $(notdir $(patsubst %.cpp,%.o, $(wildcard $(search_wildcards))))
	gcc $^ -o $@ $(additional_flags) $(link_flags)
VPATH := $(source_dirs)
%.o: %.cpp
	gcc -std=c++11 $(additional_flags) -c -MD $(compile_flags) $(addprefix -I,$(source_dirs)) $<
include $(wildcard *.d)
