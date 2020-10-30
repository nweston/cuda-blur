CUDAPATH ?= /usr/local/cuda
NVCC = ${CUDAPATH}/bin/nvcc

OPENEXR_LIBS = -lIlmImf-2_5 -lImath-2_5 -lHalf-2_5 -lIex-2_5 -lIexMath-2_5 -lIlmThread-2_5
# Note: can't use -Wpedantic here because the code CUDA passes to gcc
# triggers warnings.
WFLAGS =  -Werror -Wall

NVCCFLAGS = --compiler-options="$(WFLAGS)" -std=c++14
LINKFLAGS = -L${CUDAPATH}/lib64 -pthread -lcuda -lcudart_static -ldl -lrt

blur-test: test.o
	$(CXX) $^ -o $@ $(LINKFLAGS)

test.o: test.cu blur.cu
	$(NVCC) -c $< $(NVCCFLAGS) -I/usr/local/openexr-2.5.2/include

clean:
	rm -f *.o blur-test
