#!/bin/bash
echo "Beginning setup.."
git submodule update --init --recursive

# Download dependencies for pulp-sdk
sudo apt-get install -y build-essential git libftdi-dev libftdi1 doxygen python3-pip libsdl2-dev curl cmake libusb-1.0-0-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev
pip install --user argcomplete pyelftools

# Download gnu-gcc-toolchain
cp resources/pulp-riscv-gcc-toolchain-ubuntu22.tar.gz ./
tar -xvf pulp-riscv-gcc-toolchain-ubuntu22.tar.gz -C ./
rm pulp-riscv-gcc-toolchain-ubuntu22.tar.gz

# Set up GCC version 9.4.0
sudo apt-get update -y && \
sudo apt-get upgrade -y && \
sudo apt-get dist-upgrade -y && \
sudo apt-get install build-essential software-properties-common -y && \
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
sudo apt-get update -y && \
sudo apt-get install gcc-9 g++-9 -y && \
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
sudo update-alternatives --config gcc

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

echo "Setup successful!"
