CC := gcc -fopenmp -fpermissive -mcmodel=medium -O
CXX := g++ -fopenmp -fpermissive -mcmodel=medium -O

OBJS := lab2_io.o lab2_omp.o

all: pca

clean:
	rm -f pca $(OBJS)

pca: $(OBJS) main_omp.c
	$(CXX) $^ -o $@ -lm 

lab2_io.o: lab2_io.cpp
	$(CXX) -c -I Eigen/ $^

# force CPP compiler even for C code
# remove main if present
%.o: %.c
	$(CXX) -Dmain=local_main -c $^ -o $@

%.o: %.cpp
	$(CXX) -Dmain=local_main -c $^ -o $@

.PHONY: all clean
