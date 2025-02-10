#OLD  := MMult_cuBLAS_1
#NEW  := MMult_cuBLAS_1
NEW := MMult_cuda_3
#SMS ?= 70 75 80 86 90
SMS ?= 90

#
# sample makefile
#

CC         := nvcc 
LINKER     := $(CC)
#CFLAGS     := -O0 -g -Wall
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# CFLAGS     := -std=c++17 -O0 -g -G
CFLAGS     := -std=c++17 -O2 -I$(HOME)/local/include -I/usr/local/cuda-12.6/targets/x86_64-linux/include/ 
LDFLAGS    := -L/usr/local/cuda-12.6/targets/x86_64-linux/lib/ -L$(HOME)/local/lib -lm  -lcublas  -lopenblas -lcupti -lcuda

UTIL       := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o \
			  helper.o \
			  pm_sampling.o

TEST_OBJS  := test_MMult.o $(NEW).o 
CUPTI_OBJS  := cupti_MMult.o $(NEW).o 

%.o: %.cpp
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

%.o: %.cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

all: 
	make clean;
	make test_MMult.x
	make cupti_MMult.x
	make $(NEW).ptx

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h helper_cupti.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 
cupti_MMult.x: $(CUPTI_OBJS) $(UTIL) parameters.h helper_cupti.h
	$(LINKER) $(CUPTI_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

$(NEW).ptx: $(NEW).cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS) -ptx -c $< -o $@

run:	
	make all
	echo "version = '$(NEW)';" > outputs/output_$(NEW).m
	./test_MMult.x >> outputs/output_$(NEW).m
	#cp output_$(OLD).m output_old.m
	#cp output_$(NEW).m output_new.m

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
