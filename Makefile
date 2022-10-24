CUDAPATH ?= /usr/local/cuda
NVCC = ${CUDAPATH}/bin/nvcc

OPENEXR_LIBS = -lIlmImf-2_5 -lImath-2_5 -lHalf-2_5 -lIex-2_5 -lIexMath-2_5 -lIlmThread-2_5 -lz
# Note: can't use -Wpedantic here because the code CUDA passes to gcc
# triggers warnings.
WFLAGS =  -Werror -Wall -Wno-error=unused-function

NPP_LIBS =

EXRPATH = /usr/local/shs/openexr-2.5.2
NVCCFLAGS = --compiler-options="$(WFLAGS)" -std=c++17 -g
LINKFLAGS = -L${CUDAPATH}/lib64 -L$(EXRPATH)/lib -pthread -lcudart_static -ldl -lrt \
	${NPP_LIBS} -static-libstdc++ -static-libgcc
CXXFLAGS = -std=c++17 $(WFLAGS) -Wpedantic -I$(EXRPATH)/include -I${CUDAPATH}/include -g

blur-test: test.o exr.o
	$(CXX) $^ -o $@ $(LINKFLAGS) $(OPENEXR_LIBS)

compute-weights: compute-weights.o
	$(CXX) $^ -o $@ $(LINKFLAGS)

%.o: %.cxx
	$(CXX) -c $< $(CXXFLAGS)

test.o: test.cu blur.cu timer.h image.h weights.h
	$(NVCC) -c $< $(NVCCFLAGS)

compute-weights.o: compute-weights.cu blur.cu image.h
	$(NVCC) -c $< $(NVCCFLAGS)

clean:
	rm -f *.o blur-test compute-weights
