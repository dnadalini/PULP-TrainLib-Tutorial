USR_PATH=$(pwd)/v1.0.16-pulp-riscv-gcc-ubuntu-18
export PULP_RISCV_GCC_TOOLCHAIN=$USR_PATH
export PATH=$USR_PATH/bin:$PATH
source v1.0.16-pulp-riscv-gcc-ubuntu-18/sourceme.sh
source pulp-sdk/configs/pulp-open.sh

