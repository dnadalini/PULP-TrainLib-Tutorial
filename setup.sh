USR_PATH=$(pwd)/pulp-riscv-gcc-toolchain
export PULP_RISCV_GCC_TOOLCHAIN=$USR_PATH
export PATH=$USR_PATH/bin:$PATH
source pulp-riscv-gcc-toolchain/sourceme.sh
source pulp-sdk/configs/pulp-open.sh

