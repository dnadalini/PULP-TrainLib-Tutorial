#!/bin/bash
echo "Beginning setup.."
git submodule update --init --recursive
conda create --name trainlib-tutorial
conda activate trainlib-tutorial

# Download dependencies for pulp-sdk
sudo apt-get install -y build-essential git libftdi-dev libftdi1 doxygen python3-pip libsdl2-dev curl cmake libusb-1.0-0-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev
pip install --user argcomplete pyelftools

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

# Download dependencies for pulp-trainlib
python -m pip install argparse 
#python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install install torch torchvision torchaudio
python -m pip install torchsummary

# Checkout the correct version of PULP-TrainLib
cd pulp-trainlib
git pull origin
git checkout trainlib-tutorial
cd ..

echo "Setup successful!"
