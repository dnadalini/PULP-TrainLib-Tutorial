#!/bin/bash
echo "Beginning setup.."

# Setup repo and environment
git submodule update --init --recursive


# Download gnu-gcc-toolchain
wget https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2
tar -xf v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2 -C ./
mv v1.0.16-pulp-riscv-gcc-ubuntu-18/ pulp-riscv-gcc-toolchain/
rm -rf v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2

# Build pulp-sdk
cd pulp-riscv-gcc-toolchain
source sourceme.sh
cd ..
USR_PATH=$(pwd)/pulp-riscv-gcc-toolchain
export PULP_RISCV_GCC_TOOLCHAIN=$USR_PATH
export PATH=$USR_PATH/bin:$PATH
cd pulp-sdk/
source configs/pulp-open.sh
make build
cd ..


# Checkout the correct version of PULP-TrainLib
cd pulp-trainlib
git pull origin
git checkout trainlib-tutorial
cd ..

# Setup GVSoC's memory for this tutorial (set L1 to 256 kB)
rm pulp-sdk/rtos/pulpos/pulp/kernel/chips/pulp/link.ld
cp img/link.ld pulp-sdk/rtos/pulpos/pulp/kernel/chips/pulp/
rm pulp-sdk/tools/gap-configs/configs/chips/pulp/pulp.json
cp img/pulp.json pulp-sdk/tools/gap-configs/configs/chips/pulp/

echo "Setup successful!"
