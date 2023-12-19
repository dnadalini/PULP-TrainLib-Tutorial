#!/bin/bash
echo "Beginning setup.."

# Setup a conda environment

# Download dependencies for pulp-sdk

# Download dependencies for pulp-trainlib

# Download gnu-gcc-toolchain
wget https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2
tar -xvzf v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2
rm -rf v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2

# Build pulp-sdk

echo "Setup successful!"
