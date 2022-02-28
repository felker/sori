NVCC=/usr/local/cuda/bin/nvcc
INC=-I/usr/local/cuda/include -I/usr/include/
NVCC_FLAGS_M=-g -G -Xcompiler -Wall -lrt -lcurand -lhdf5 -L/usr/local/cuda/lib64
NVCC_FLAGS=-g -G -Xcompiler -Wall -lrt -lcurand -lhdf5 -L/usr/local/cuda/lib64 -dlink -gencode arch=compute_70,code=sm_70

all: sori.exe

sori.exe: sori.o sori_kernel.o dprf.o dprf_kernel.o
	$(NVCC) $(NVCC_FLAGS_M) $^ -o $@
	
sori.o: sori.cpp sori_kernel.h
	$(NVCC) $(NVCC_FLAGS_M) -c $< -o $@

sori_kernel.o: sori_kernel.cu sori_kernel.h
	$(NVCC) $(NVCC_FLAGS_M) -c $< -o $@

dprf_test.exe: dprf_test.o dprf.o dprf_kernel.o
	$(NVCC) $(NVCC_FLAGS_M) $^ -o $@
	
dprf_test.o: dprf_test.c dprf.h dprf_kernel.o
	$(NVCC) $(NVCC_FLAGS_M) -c $< -o $@
			
dprf.o: dprf.c dprf.h dprf_kernel.h
	$(NVCC) $(NVCC_FLAGS_M) -c $< -o $@
	
dprf_kernel.o: dprf_kernel.cu dprf_kernel.h
	$(NVCC) $(NVCC_FLAGS_M) -c $< -o $@

cubin: sori_kernel.cu dprf_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -cubin sori_kernel.cu -o sori_kernel.cubin
	$(NVCC) $(NVCC_FLAGS) -cubin dprf_kernel.cu -o dprf_kernel.cubin
	
clean:
	rm -rf *.o *.exe *.cubin __pycache__	
