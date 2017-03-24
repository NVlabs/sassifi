# Setup your variables in env.mk before building!
include ../../env.mk

# Decide which libraries you want to compile.
export LIBS := 	libprofiler.a \
		librfinjector.a \
		libinstvalueinjector.a \
		libinstaddressinjector.a 

NVCC = $(CUDA_HOME)/bin/nvcc
CUPTI = $(CUDA_HOME)/extras/CUPTI

ifndef INSTALL_DIR
	INSTALL_DIR := ../..
endif

INSTALLED_LIBS := $(addprefix $(INSTALL_DIR)/lib/, $(LIBS))

########################################
# Flags that are useful for debugging. 
########################################

# Instrumentation handler will be empty if EMPTY_HANDLER=1. 
EMPTY_HANDLER=0

# During error injection run, the error will not be injected if this
# DUMMY_INJECTION=1. The rest of the code will execute, including the functions
# that are written to inject the error. Just the c-code that perturbs the value
# will be turned off. 
DUMMY_INJECTION=0

# Print timing per kernel launch
TIMING=0

# Debug flags
INJ_DEBUG_LIGHT=0
INJ_DEBUG_HEAVY=0

# Flags used by the build commands below
RF_INJECTIONS=0
INST_VALUE_INJECTIONS=1
INST_ADDRESS_INJECTIONS=2

FLAGS = -ccbin $(CCBIN) -std=c++11 \
		--compiler-options -Wall \
		-O3 $(GENCODE) \
		-lineinfo -c -rdc=true  \
		--maxrregcount=16 \
		--compiler-options -Wall \
		-DEMPTY_HANDLER=$(EMPTY_HANDLER) \
		-DDUMMY_INJECTION=$(DUMMY_INJECTION) \
		-DTIMING=$(TIMING) \
		-DINJ_DEBUG_LIGHT=$(INJ_DEBUG_LIGHT) \
		-DINJ_DEBUG_HEAVY=$(INJ_DEBUG_HEAVY) \
		-I$(CUPTI)/include/ \
		-I$(INSTALL_DIR)/utils/ \
		-I$(SASSI_HOME)/include/ \


########################################
# build rules: try "make -j4"
########################################

install: $(INSTALLED_LIBS)

$(INSTALL_DIR)/lib/lib%.a: %.o
	ar r $@ $^
	ranlib $@

profiler.o: profiler.cu
	$(NVCC) $(FLAGS) -o $@ $^ 

rfinjector.o: injector.cu 
	$(NVCC) $(FLAGS) -o $@ $^ \
		-DINJ_MODE=$(RF_INJECTIONS)

instvalueinjector.o: injector.cu
	$(NVCC) $(FLAGS) -o $@ $^ \
		-DINJ_MODE=$(INST_VALUE_INJECTIONS)

instaddressinjector.o: injector.cu
	$(NVCC) $(FLAGS) -o $@ $^ \
		-DINJ_MODE=$(INST_ADDRESS_INJECTIONS)

clean:
	$(RM) $(INSTALLED_LIBS) *.o *.a *~ 
