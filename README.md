# PULP-TrainLib-Tutorial

This repository contains several tutorial sessions to becom familiar with PULP-TrainLib and On-Device Learning on RISC-V Parallel Ultra-Low Power systems.

## Prerequisites

This tutorial has been tested with the following requirements:
- Ubuntu <= 20.04 LTS
- Anaconda or Miniconda
- make
- GCC <= 9.4

On other setups, we do not provide any guatantee.

## Installation

To install all the components required by PULP-TrainLib, simply:
- source install_ub18.sh

Then, close the installation terminal and open a new one.

An experimental setup for Ubuntu 22 is provided, but not guarantee to work.

## Executing tests of PULP-TrainLib

To compile and run PULP-TrainLib's code:

- source setup.sh to set up PULP-SDK
- cd pulp-trainlib/tests/xxx
- make clean get_golden all run <OPTIONS>

You can check the available options into the Makefile of each test.
