# lavm 

A massively parallel virtual machine based on the [RASP architecture](https://en.wikipedia.org/wiki/Random-access_stored-program_machine).

### Experiments

To reproduce the JVM experiments, run the following command (tested on M4 Max):

```agsl
./gradlew run
```

This will run a parallel streaming search for heapless programs and print out the longest-running busy beavers.

To reproduce the CUDA experiments, first generate the VMs like so:

```
./gradlew writeVMs
```

Then either (1) run the following commands on hardware with Compute Capability 7.5 or higher, e.g., for an H100:

```agsl
cd cuda

nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v -gencode arch=compute_90,code=sm_90 rasp_cuda_benchmark_soa.cu -o rasp_cuda_benchmark_soa
  
./rasp_cuda_benchmark_soa rasp_vms.bin
```

or (2) deploy remotely. We provide a Modal harness [here](cuda/rasp_cuda_modal_cu_soa.py).

### Paper

* [https://arxiv.org/pdf/2604.12902](https://arxiv.org/pdf/2604.12902)