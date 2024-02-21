APP = linear_test_fp16

# User settings
IN_CH?=128
OUT_CH?=128
NUM_CORES?=1
STEP?='BACKWARD_ERROR' # Possible steps: 'FORWARD', 'BACKWARD_GRAD', 'BACKWARD_ERROR'
MATMUL_TYPE?=0
TRANSPOSE_WEIGHTS?=0	# Choose if to save the weights as transposed or not
# End of user settings

TRAIN_LIB=./lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_vector_matrix_fp16.c

DATA_TYPE?='fp16'
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3 
APP_CFLAGS += -DFABRIC 
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DMEMOCC_COMP
APP_CFLAGS += -mhwloopalign
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}
APP_CFLAGS += -DTRANSPOSE_WEIGHTS=$(TRANSPOSE_WEIGHTS)
APP_CFLAGS += -DOPTIMIZE
#APP_CFLAGS += -DDEBUG
APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	python3 utils/GM.py --in_size $(IN_CH) --out_size $(OUT_CH) --step $(STEP) --transpose $(TRANSPOSE_WEIGHTS)

include $(RULES_DIR)/pmsis_rules.mk
