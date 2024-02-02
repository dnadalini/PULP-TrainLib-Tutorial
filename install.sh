#!/bin/bash
echo "Beginning setup.."
git submodule update --init --recursive

# Download dependencies for pulp-sdk
sudo apt-get install -y build-essential git libftdi-dev libftdi1 doxygen python3-pip libsdl2-dev curl cmake libusb-1.0-0-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev
pip install --user argcomplete pyelftools

# Download gnu-gcc-toolchain
wget https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2
tar -xf v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2 v1.0.16-pulp-riscv-gcc-ubuntu-18
rm -rf v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2

# Build pulp-sdk
cd v1.0.16-pulp-riscv-gcc-ubuntu-18
source sourceme.sh
cd ..
USR_PATH=$(pwd)/v1.0.16-pulp-riscv-gcc-ubuntu-18/
export PULP_RISCV_GCC_TOOLCHAIN=$USR_PATH
cd pulp-sdk/
source configs/pulp-open.sh
make build

# Download dependencies for pulp-trainlib
python -m pip install argparse 
python -m pip install install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install torchsummary

echo "Setup successful!"
