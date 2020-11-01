CUDAPATH ?= /usr/local/cuda
NVCC = ${CUDAPATH}/bin/nvcc

OPENEXR_LIBS = -lIlmImf-2_5 -lImath-2_5 -lHalf-2_5 -lIex-2_5 -lIexMath-2_5 -lIlmThread-2_5 -lz
# Note: can't use -Wpedantic here because the code CUDA passes to gcc
# triggers warnings.
WFLAGS =  -Werror -Wall

EXRPATH = /usr/local/openexr-2.5.2
NVCCFLAGS = --compiler-options="$(WFLAGS)" -std=c++17
LINKFLAGS = -L${CUDAPATH}/lib64 -L$(EXRPATH)/lib -pthread -lcuda -lcudart_static -ldl -lrt
CXXFLAGS = -std=c++17 $(WFLAGS) -Wpedantic -I$(EXRPATH)/include -I${CUDAPATH}/include

blur-test: test.o exr.o
	$(CXX) $^ -o $@ $(LINKFLAGS) $(OPENEXR_LIBS)

exr.o: exr.cxx
	$(CXX) -c $< $(CXXFLAGS)

test.o: test.cu blur.cu
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f *.o blur-test
