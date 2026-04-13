#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int MAX_BENCHMARK_VMS = 8000000;
constexpr int MEM_WORDS = 128;

// Keep old on-disk layout compatible:
constexpr int SERIALIZED_OUTPUT_WORDS = 3;

// But let compiled programs emit up to 9 outputs at runtime:
constexpr int OUTPUT_WORDS = 9;

constexpr int MAX_STEPS = 1000000;

constexpr int THREADS_PER_BLOCK = 256;
constexpr int HOT_STATE_WORDS = 5; // pc, acc, steps, halted, outCount
constexpr int FILE_PREFIX_WORDS = HOT_STATE_WORDS + SERIALIZED_OUTPUT_WORDS;

constexpr uint32_t OPCODE_BITS = 3u;
constexpr uint32_t OPCODE_MASK = (1u << OPCODE_BITS) - 1u; // 0b111
constexpr uint32_t WORD_BITS = 29u;
constexpr uint32_t WORD_MASK = (1u << WORD_BITS) - 1u;     // 0x1FFFFFFF

#define CUDA_CHECK(stmt)                                                        \
  do {                                                                          \
    cudaError_t err__ = (stmt);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string(#stmt) + " failed: " +               \
                               cudaGetErrorString(err__));                      \
    }                                                                           \
  } while (0)

__device__ __forceinline__ uint32_t wrap(uint32_t x, uint32_t n) { return x % n; }

__device__ __forceinline__ uint32_t norm_word(uint32_t x) { return x & WORD_MASK; }

__device__ __forceinline__ uint32_t opcode_of(uint32_t ins) { return ins & OPCODE_MASK; }

__device__ __forceinline__ uint32_t operand_of(uint32_t ins) { return ins >> OPCODE_BITS; }

__device__ __forceinline__ uint32_t add_word(uint32_t x, uint32_t y) { return (x + y) & WORD_MASK; }

__device__ __forceinline__ uint32_t mul_word(uint32_t x, uint32_t y) {
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(x & WORD_MASK) *
       static_cast<uint64_t>(y & WORD_MASK)) &
      static_cast<uint64_t>(WORD_MASK));
}

__global__ void run_hypervisor(
    uint32_t* __restrict__ pcs,
    uint32_t* __restrict__ accs,
    uint32_t* __restrict__ stepss,
    uint32_t* __restrict__ halteds,
    uint32_t* __restrict__ out_counts,
    uint32_t* __restrict__ outputs,
    uint32_t* __restrict__ mems,
    uint32_t vm_count,
    uint32_t max_steps,
    uint32_t mem_words,
    uint32_t output_words) {
  uint32_t vm = blockIdx.x * blockDim.x + threadIdx.x;
  if (vm >= vm_count) return;

  uint32_t pc = pcs[vm];
  uint32_t acc = accs[vm] & WORD_MASK;
  uint32_t steps = stepss[vm];
  uint32_t halted = halteds[vm];
  uint32_t out_count = out_counts[vm];

  const uint32_t mem_base = vm * mem_words;
  const uint32_t out_base = vm * output_words;

  while (halted == 0u && steps < max_steps) {
    const uint32_t ins = mems[mem_base + wrap(pc, mem_words)];
    const uint32_t op = opcode_of(ins);
    const uint32_t arg = operand_of(ins);

    switch (op) {
      case 1u: // LOD imm
        acc = norm_word(arg);
        pc += 1u;
        break;

      case 2u: // ADD addr
        acc = add_word(acc, mems[mem_base + wrap(arg, mem_words)]);
        pc += 1u;
        break;

      case 3u: // MUL addr
        acc = mul_word(acc, mems[mem_base + wrap(arg, mem_words)]);
        pc += 1u;
        break;

      case 4u: // STO addr
        mems[mem_base + wrap(arg, mem_words)] = norm_word(acc);
        pc += 1u;
        break;

      case 5u: // BNZ label
        pc = (acc != 0u) ? arg : (pc + 1u);
        break;

      case 7u: { // PRI addr
        const uint32_t value = mems[mem_base + wrap(arg, mem_words)];
        if (out_count < output_words) {
          outputs[out_base + out_count] = value;
          ++out_count;
        }
        pc += 1u;
        break;
      }

      default: // invalid packed instruction = HLT
        halted = 1u;
        break;
    }

    ++steps;
  }

  if (steps >= max_steps) halted = 1u;

  pcs[vm] = pc;
  accs[vm] = norm_word(acc);
  stepss[vm] = steps;
  halteds[vm] = halted;
  out_counts[vm] = out_count;
}

struct HostState {
  std::vector<uint32_t> pcs;
  std::vector<uint32_t> accs;
  std::vector<uint32_t> stepss;
  std::vector<uint32_t> halteds;
  std::vector<uint32_t> out_counts;
  std::vector<uint32_t> outputs; // runtime output tape
  std::vector<uint32_t> mems;
};

static uint32_t read_be_u32(std::istream& in) {
  unsigned char b[4];
  if (!in.read(reinterpret_cast<char*>(b), 4)) {
    throw std::runtime_error("Unexpected EOF while reading VM state");
  }
  return (static_cast<uint32_t>(b[0]) << 24) |
         (static_cast<uint32_t>(b[1]) << 16) |
         (static_cast<uint32_t>(b[2]) <<  8) |
         (static_cast<uint32_t>(b[3]) <<  0);
}

HostState load_vms(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Could not open " + path);

  constexpr size_t INTS_PER_VM =
      static_cast<size_t>(FILE_PREFIX_WORDS) + MEM_WORDS + 1;
  const size_t expected_bytes =
      static_cast<size_t>(MAX_BENCHMARK_VMS) * INTS_PER_VM * sizeof(uint32_t);

  in.seekg(0, std::ios::end);
  const size_t actual_bytes = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);

  if (actual_bytes != expected_bytes) {
    throw std::runtime_error(
        "File size mismatch for " + path +
        ": expected " + std::to_string(expected_bytes) +
        " bytes, found " + std::to_string(actual_bytes));
  }

  HostState s{
      std::vector<uint32_t>(MAX_BENCHMARK_VMS),
      std::vector<uint32_t>(MAX_BENCHMARK_VMS),
      std::vector<uint32_t>(MAX_BENCHMARK_VMS),
      std::vector<uint32_t>(MAX_BENCHMARK_VMS),
      std::vector<uint32_t>(MAX_BENCHMARK_VMS),
      std::vector<uint32_t>(static_cast<size_t>(MAX_BENCHMARK_VMS) * OUTPUT_WORDS, 0u),
      std::vector<uint32_t>(static_cast<size_t>(MAX_BENCHMARK_VMS) * MEM_WORDS),
  };

  for (int vm = 0; vm < MAX_BENCHMARK_VMS; ++vm) {
    for (int off = 0; off < FILE_PREFIX_WORDS; ++off) {
      const uint32_t word = read_be_u32(in);

      switch (off) {
        case 0: s.pcs[vm]        = word; break;
        case 1: s.accs[vm]       = word; break;
        case 2: s.stepss[vm]     = word; break;
        case 3: s.halteds[vm]    = word; break;
        case 4: s.out_counts[vm] = word; break;
        default:
          s.outputs[static_cast<size_t>(vm) * OUTPUT_WORDS + (off - HOT_STATE_WORDS)] = word;
          break;
      }
    }

    const size_t base = static_cast<size_t>(vm) * MEM_WORDS;
    for (int i = 0; i < MEM_WORDS; ++i) {
      s.mems[base + i] = read_be_u32(in);
    }

    const uint32_t terminator = read_be_u32(in);
    if (terminator != static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error(
          "VM " + std::to_string(vm) +
          ": expected terminator " +
          std::to_string(std::numeric_limits<int32_t>::max()) +
          ", got " + std::to_string(terminator));
    }
  }

  return s;
}

void print_halting_distribution(
    const std::vector<uint32_t>& stepss,
    const std::vector<uint32_t>& halteds,
    uint32_t vm_count) {
  constexpr uint32_t HALT_EXACT_MAX = 100;
  constexpr uint32_t HALT_OVERFLOW_INDEX = HALT_EXACT_MAX;
  std::vector<uint64_t> buckets(HALT_EXACT_MAX + 1, 0);

  uint64_t timed_out = 0;

  for (uint32_t vm = 0; vm < vm_count; ++vm) {
    if (halteds[vm] == 0u) continue;
    const uint32_t steps = stepss[vm];
    if (steps == 0u) continue;

    if (steps >= MAX_STEPS) {
      ++timed_out;
      continue;
    }

    if (steps <= HALT_EXACT_MAX) ++buckets[steps - 1u];
    else ++buckets[HALT_OVERFLOW_INDEX];
  }

  std::cout << "Halting distribution (excluding >= " << MAX_STEPS << "-step timeouts):\n";
  for (uint32_t steps = 1; steps <= HALT_EXACT_MAX; ++steps) {
    std::cout << "\t(" << steps << "," << buckets[steps - 1u] << ")\n";
  }
  std::cout << "\t(101+," << buckets[HALT_OVERFLOW_INDEX] << ")\n";
  std::cout << "Excluded timed-out VMs: " << timed_out << "\n";
}

struct Shard {
  uint32_t offset = 0;
  uint32_t count = 0;
};

static std::vector<Shard> split_evenly(uint32_t vm_count, int parts) {
  std::vector<Shard> shards(parts);
  const uint32_t base = vm_count / static_cast<uint32_t>(parts);
  const uint32_t rem  = vm_count % static_cast<uint32_t>(parts);

  uint32_t offset = 0;
  for (int i = 0; i < parts; ++i) {
    const uint32_t count = base + (static_cast<uint32_t>(i) < rem ? 1u : 0u);
    shards[i] = Shard{offset, count};
    offset += count;
  }
  return shards;
}

struct DeviceBuffers {
  int device = -1;
  uint32_t capacity = 0;

  uint32_t* pcs = nullptr;
  uint32_t* accs = nullptr;
  uint32_t* stepss = nullptr;
  uint32_t* halteds = nullptr;
  uint32_t* out_counts = nullptr;
  uint32_t* outputs = nullptr;
  uint32_t* mems = nullptr;

  DeviceBuffers() = default;
  DeviceBuffers(const DeviceBuffers&) = delete;
  DeviceBuffers& operator=(const DeviceBuffers&) = delete;
  DeviceBuffers(DeviceBuffers&&) = delete;
  DeviceBuffers& operator=(DeviceBuffers&&) = delete;

  void allocate(int dev, uint32_t cap) {
    device = dev;
    capacity = cap;
    if (capacity == 0) return;

    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMalloc(&pcs,        static_cast<size_t>(capacity) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&accs,       static_cast<size_t>(capacity) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&stepss,     static_cast<size_t>(capacity) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&halteds,    static_cast<size_t>(capacity) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out_counts, static_cast<size_t>(capacity) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&outputs,    static_cast<size_t>(capacity) * OUTPUT_WORDS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&mems,       static_cast<size_t>(capacity) * MEM_WORDS * sizeof(uint32_t)));
  }

  void release() noexcept {
    if (device < 0) return;
    cudaSetDevice(device);
    if (pcs)        cudaFree(pcs);
    if (accs)       cudaFree(accs);
    if (stepss)     cudaFree(stepss);
    if (halteds)    cudaFree(halteds);
    if (out_counts) cudaFree(out_counts);
    if (outputs)    cudaFree(outputs);
    if (mems)       cudaFree(mems);

    pcs = accs = stepss = halteds = out_counts = outputs = mems = nullptr;
    capacity = 0;
  }

  ~DeviceBuffers() {
    release();
  }
};

static void upload_shard(
    const HostState& master,
    DeviceBuffers& d,
    const Shard& shard) {
  if (shard.count == 0) return;
  if (shard.count > d.capacity) {
    throw std::runtime_error("Shard larger than allocated device capacity");
  }

  CUDA_CHECK(cudaSetDevice(d.device));

  const size_t vm_bytes  = static_cast<size_t>(shard.count) * sizeof(uint32_t);
  const size_t out_bytes = static_cast<size_t>(shard.count) * OUTPUT_WORDS * sizeof(uint32_t);
  const size_t mem_bytes = static_cast<size_t>(shard.count) * MEM_WORDS * sizeof(uint32_t);

  const size_t out_off = static_cast<size_t>(shard.offset) * OUTPUT_WORDS;
  const size_t mem_off = static_cast<size_t>(shard.offset) * MEM_WORDS;

  CUDA_CHECK(cudaMemcpy(d.pcs,        master.pcs.data()        + shard.offset, vm_bytes,  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.accs,       master.accs.data()       + shard.offset, vm_bytes,  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.stepss,     master.stepss.data()     + shard.offset, vm_bytes,  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.halteds,    master.halteds.data()    + shard.offset, vm_bytes,  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.out_counts, master.out_counts.data() + shard.offset, vm_bytes,  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.outputs,    master.outputs.data()    + out_off,      out_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.mems,       master.mems.data()       + mem_off,      mem_bytes, cudaMemcpyHostToDevice));
}

static void download_shard_results(
    const DeviceBuffers& d,
    const Shard& shard,
    std::vector<uint32_t>& final_stepss,
    std::vector<uint32_t>& final_halteds,
    std::vector<uint32_t>& final_out_counts,
    std::vector<uint32_t>& final_outputs) {
  if (shard.count == 0) return;

  CUDA_CHECK(cudaSetDevice(d.device));

  const size_t vm_bytes  = static_cast<size_t>(shard.count) * sizeof(uint32_t);
  const size_t out_bytes = static_cast<size_t>(shard.count) * OUTPUT_WORDS * sizeof(uint32_t);
  const size_t out_off   = static_cast<size_t>(shard.offset) * OUTPUT_WORDS;

  CUDA_CHECK(cudaMemcpy(final_stepss.data()     + shard.offset, d.stepss,     vm_bytes,  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(final_halteds.data()    + shard.offset, d.halteds,    vm_bytes,  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(final_out_counts.data() + shard.offset, d.out_counts, vm_bytes,  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(final_outputs.data()    + out_off,      d.outputs,    out_bytes, cudaMemcpyDeviceToHost));
}

int main(int argc, char** argv) {
  try {
    const std::string path = argc > 1 ? argv[1] : "rasp_vms.bin";
    HostState master = load_vms(path);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
      throw std::runtime_error("No CUDA devices visible");
    }

    std::cerr << "Visible CUDA devices: " << device_count << "\n";
    for (int dev = 0; dev < device_count; ++dev) {
      cudaDeviceProp prop{};
      CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
      std::cerr << "GPU " << dev << ": " << prop.name << "\n";
    }

    const uint32_t max_shard_capacity =
        (MAX_BENCHMARK_VMS + static_cast<uint32_t>(device_count) - 1u) /
        static_cast<uint32_t>(device_count);

    std::vector<DeviceBuffers> bufs(static_cast<size_t>(device_count));
    for (int dev = 0; dev < device_count; ++dev) {
      bufs[dev].allocate(dev, max_shard_capacity);
    }

    for (int x = 1; x <= 8; ++x) {
      const uint32_t vm_count = static_cast<uint32_t>(x) * 1000000u;
      const auto shards = split_evenly(vm_count, device_count);

      // Upload each shard to its GPU.
      for (int dev = 0; dev < device_count; ++dev) {
        upload_shard(master, bufs[dev], shards[dev]);
      }

      // Ensure all H2D copies are done before timing.
      for (int dev = 0; dev < device_count; ++dev) {
        if (shards[dev].count == 0) continue;
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      // Launch one shard per GPU. These kernels run concurrently across devices.
      const auto start = std::chrono::steady_clock::now();

      for (int dev = 0; dev < device_count; ++dev) {
        const uint32_t count = shards[dev].count;
        if (count == 0) continue;

        CUDA_CHECK(cudaSetDevice(dev));
        const int blocks = static_cast<int>((count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);

        run_hypervisor<<<blocks, THREADS_PER_BLOCK>>>(
            bufs[dev].pcs,
            bufs[dev].accs,
            bufs[dev].stepss,
            bufs[dev].halteds,
            bufs[dev].out_counts,
            bufs[dev].outputs,
            bufs[dev].mems,
            count,
            MAX_STEPS,
            MEM_WORDS,
            OUTPUT_WORDS);
        CUDA_CHECK(cudaGetLastError());
      }

      // Wait for all GPUs.
      for (int dev = 0; dev < device_count; ++dev) {
        if (shards[dev].count == 0) continue;
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      const auto end = std::chrono::steady_clock::now();
      const double secs = std::chrono::duration<double>(end - start).count();

      std::cout << "(" << x << "," << std::fixed << std::setprecision(1) << secs << ")\n";

      if (vm_count == MAX_BENCHMARK_VMS) {
        std::vector<uint32_t> final_stepss(vm_count);
        std::vector<uint32_t> final_halteds(vm_count);
        std::vector<uint32_t> final_out_counts(vm_count);
        std::vector<uint32_t> final_outputs(static_cast<size_t>(vm_count) * OUTPUT_WORDS);

        for (int dev = 0; dev < device_count; ++dev) {
          download_shard_results(
              bufs[dev],
              shards[dev],
              final_stepss,
              final_halteds,
              final_out_counts,
              final_outputs);
        }

        print_halting_distribution(final_stepss, final_halteds, vm_count);

        // tiny sanity dump for global VMs 0,1,2
        for (int vm = 0; vm < 3; ++vm) {
          std::cout << "vm=" << vm << " out=[";
          for (uint32_t j = 0; j < final_out_counts[vm] && j < OUTPUT_WORDS; ++j) {
            if (j) std::cout << ",";
            std::cout << final_outputs[static_cast<size_t>(vm) * OUTPUT_WORDS + j];
          }
          std::cout << "]\n";
        }
      }
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}