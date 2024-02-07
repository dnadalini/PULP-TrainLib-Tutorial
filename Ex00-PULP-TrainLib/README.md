# PULP-TrainLib: a General Overview

This introductory tutorial aims at presenting the structure and key features of PULP-TrainLib. 

PULP-TrainLib is the first On-Device Learning optimized for RISC-V Multi-Core MCUs, tailored for the Parallel Ultra-Low Power [(PULP) Platform](https://www.pulp-platform.org/). More specifically, PULP-TrainLib makes efficient use of the available resources of the PULP.based SoCs, which feature:

- A single Core (Fabric Controller) for the control of the system
- A Cluster of N RISC-V Cores capable of computing parallel tasks
- A hierarchical memory system, featuring a Cluster-reserved fast L1 memory and a system-level L2 memory
- A Cluster DMA to access the L2 memory from the Cluster in few cycles
- Tightly coupled accelerators, as a Mixed-Precision FPU, available for the Cluster 

An example of a PULP-based SoC is the following:

![PULP](../img/PULP.png)

In this embodiment, the Cluster features 8 parallel cores (Cores 0 to 7) for the computation, and a Cluster Controller core (Core 8) to better schedule tasks assigned to the Cluster.

## General Structure

PULP-TrainLib is available as open source [here](https://github.com/pulp-platform/pulp-trainlib). Further details on PULP-TrainLib can be found in the related [README.md](../pulp-trainlib/README.md). 

Briefly, PULP-TrainLib 

```
Here, talk about what the different folders contain and give some info.
```

## Launching a Test

## Contributing

## References

> D. Nadalini, M. Rusci, G. Tagliavini, L. Ravaglia, L. Benini, and F. Conti, "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning" [SAMOS Pre-Print Version](https://www.samos-conference.com/Resources_Samos_Websites/Proceedings_Repository_SAMOS/2022/Papers/Paper_14.pdf), [Springer Published Version](https://link.springer.com/chapter/10.1007/978-3-031-15074-6_13)

> D. Nadalini, M. Rusci, L. Benini, and F. Conti, "Reduced Precision Floating-Point Optimization for Deep Neural Network On-Device Learning on MicroControllers" [ArXiv Pre-Print](https://arxiv.org/abs/2305.19167)
