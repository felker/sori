PROG := soriRt

CUDA := /usr/local/cuda

NVCC := nvcc

MARCH := $(if $(findstring x86, $(MAKE_HOST)), -march=native,)
CPPFLAGS := -I$(CUDA)/include
CXXFLAGS := -std=gnu++17 -g -Wall -O3 $(MARCH)
LDFLAGS := -L$(CUDA)/lib64
LDLIBS := -lcudart -lhdf5

OBJ := util/logger.o
OBJ += rth5reader.o
OBJ += rtgpu_sori.o
OBJ += rtgpu_sori_kernel.o
OBJ += rtgpu_dprf_kernel.o
OBJ += soriRt.o

.PHONY: all
all: $(PROG)

$(PROG): $(OBJ)
	$(LINK.cc) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

%.o: %.cu
	$(NVCC) -arch=native -O3 -c $< -o $@

.PHONY: clean
clean:
	$(RM) $(PROG) *.o */*.o
