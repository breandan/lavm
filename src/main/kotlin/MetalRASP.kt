package org.example

import ai.hypergraph.kaliningraph.automata.completeWithSparseGRE
import ai.hypergraph.kaliningraph.parsing.tmLst
import com.sun.jna.Library
import com.sun.jna.Native
import java.io.File
import kotlin.system.measureNanoTime
import kotlin.time.TimeSource

private const val MR_DEFAULT_RASP_VMS = METAL_RASP_VMS
private const val MR_DEFAULT_MAX_CONCURRENT_THREADS = MAX_CONCURRENT_THREADS
private const val MR_MEM_WORDS = MEM_WORDS
private const val MR_OUTPUT_WORDS = OUTPUT_WORDS
private const val MR_DEFAULT_QUANTUM = QUANTUM
private const val MR_DEFAULT_MAX_STEPS = MAX_STEPS
private const val MR_MEM_BASE = MEM_BASE
private const val MR_VM_STRIDE = VM_STRIDE

/*
./gradlew metalRASP
*/
fun main() {
  val vmCount = intSetting("rasp.vmCount", "RASP_VM_COUNT", MR_DEFAULT_RASP_VMS)
  val workers = intSetting("rasp.workers", "RASP_WORKERS", MR_DEFAULT_MAX_CONCURRENT_THREADS)
  val quantum = intSetting("rasp.quantum", "RASP_QUANTUM", MR_DEFAULT_QUANTUM)
  val maxSteps = intSetting("rasp.maxSteps", "RASP_MAX_STEPS", MR_DEFAULT_MAX_STEPS)
  require(vmCount > 0) { "VM count must be positive" }
  require(workers > 0) { "Worker count must be positive" }
  require(quantum > 0) { "Quantum must be positive" }
  require(maxSteps > 0) { "Max steps must be positive" }

  val cfg = loopyHeapless
  val vmWords = IntArray(vmCount * MR_VM_STRIDE)
  val programSources = arrayOfNulls<String>(vmCount)
  val packTimer = TimeSource.Monotonic.markNow()
  completeWithSparseGRE(List(100) { "_" }, cfg)!!
    .sampleStrWithoutReplacement(cfg.tmLst)
    .take(vmCount)
    .forEachIndexed { vm, source ->
      programSources[vm] = source
      packProgramIntoVm(source.compileToRASPBytecode(), vmWords, vm)
    }

  println("Packed $vmCount RASP programs in ${packTimer.elapsedNow()}")

  val elapsedNs = measureNanoTime {
    val ok = metalBridge().metalRunHypervisor(
      RASP_MSL,
      "run_hypervisor",
      vmCount,
      workers,
      MR_MEM_WORDS,
      MR_OUTPUT_WORDS,
      quantum,
      maxSteps,
      vmWords
    )
    require(ok != 0) { "metalRunHypervisor failed" }
  }

  val halted = countMetalHalted(vmWords, vmCount)
  printStepHistogram(vmWords, vmCount, maxSteps, bucketSize = 10)
  printLongestRunningPrograms(vmWords, programSources, vmCount, maxSteps, topK = 10)
  val secs = elapsedNs / 1e9
  val rate = if (secs > 0.0) vmCount / secs else Double.POSITIVE_INFINITY
  println("programs=$vmCount halted=$halted elapsed=${secs}s rate=${rate} prog/s")
}

private fun intSetting(propertyName: String, envName: String, default: Int): Int =
  System.getProperty(propertyName)?.toIntOrNull()
    ?: System.getenv(envName)?.toIntOrNull()
    ?: default

private fun packProgramIntoVm(program: IntArray, vmWords: IntArray, vm: Int) {
  require(program.size <= MR_MEM_WORDS) { "Program has ${program.size} packed instructions, exceeds VM memory budget $MR_MEM_WORDS" }
  program.copyInto(vmWords, vm * MR_VM_STRIDE + MR_MEM_BASE)
}

private fun countMetalHalted(vmWords: IntArray, vmCount: Int): Int {
  var n = 0
  var vm = 0
  while (vm < vmCount) {
    if (vmWords[vm * MR_VM_STRIDE + HALTED] != 0) n++
    vm++
  }
  return n
}

private fun printStepHistogram(vmWords: IntArray, vmCount: Int, maxSteps: Int, bucketSize: Int) {
  require(bucketSize > 0) { "bucketSize must be positive" }

  val counts = linkedMapOf<Int, Int>()
  var timedOut = 0
  var nonHalted = 0
  var vm = 0

  while (vm < vmCount) {
    val base = vm * MR_VM_STRIDE
    val halted = vmWords[base + HALTED] != 0
    val steps = vmWords[base + STEPS]

    when {
      !halted -> nonHalted++
      steps >= maxSteps -> timedOut++
      else -> {
        val bucketStart = (steps / bucketSize) * bucketSize
        counts[bucketStart] = (counts[bucketStart] ?: 0) + 1
      }
    }
    vm++
  }

  println("natural-halt histogram (bucketSize=$bucketSize):")
  for ((start, count) in counts.toSortedMap()) {
    println("  ${start}-${start + bucketSize}: $count")
  }
  if (timedOut > 0) println("  timed-out-at-$maxSteps: $timedOut")
  if (nonHalted > 0) println("  non-halted: $nonHalted")
}

private fun printLongestRunningPrograms(vmWords: IntArray, programSources: Array<String?>, vmCount: Int, maxSteps: Int, topK: Int) {
  require(topK > 0) { "topK must be positive" }

  val top = ArrayList<MetalHaltCandidate>(topK)
  var vm = 0
  while (vm < vmCount) {
    val base = vm * MR_VM_STRIDE
    val halted = vmWords[base + HALTED] != 0
    val steps = vmWords[base + STEPS]
    val naturallyHalted = halted && steps < maxSteps

    if (naturallyHalted) {
      if (top.size < topK) {
        top += MetalHaltCandidate(vm, steps)
        top.sortBy { it.steps }
      } else if (steps > top[0].steps) {
        top[0] = MetalHaltCandidate(vm, steps)
        top.sortBy { it.steps }
      }
    }
    vm++
  }

  println("top $topK longest-running naturally halting programs:")
  for ((rank, candidate) in top.sortedByDescending { it.steps }.withIndex()) {
    val source = programSources[candidate.vm] ?: "<missing source>"
    println("  ${rank + 1}.) steps=${candidate.steps} vm=${candidate.vm}")
    println(source)
  }
}

private data class MetalHaltCandidate(val vm: Int, val steps: Int)

interface MetalBridge : Library {
  fun metalRunHypervisor(
    mslSource: String,
    kernelName: String,
    vmCount: Int,
    workers: Int,
    memWords: Int,
    outputWords: Int,
    quantum: Int,
    maxSteps: Int,
    vmWords: IntArray,
  ): Int
}

internal fun metalBridge(): MetalBridge {
  if (!System.getProperty("os.name").startsWith("Mac")) throw Exception("Only works on MacOS")
  val directory = File("build/metalRasp").also { it.mkdirs() }
  val swiftFile = directory.resolve("RASPHypervisorBridge.swift")
  val dylib = directory.resolve("libRASPHypervisorBridge.dylib")
  val hashFile = directory.resolve(".raspHypervisorHash")
  val swiftSrc = metalBridgeSwiftSource

  if (!dylib.exists() || !hashFile.exists() || hashFile.readText() != swiftSrc.hashCode().toString()) {
    swiftFile.writeText(swiftSrc)
    val exit = ProcessBuilder(
      "xcrun", "swiftc", "-O", "-emit-library",
      swiftFile.absolutePath,
      "-o", dylib.absolutePath,
      "-module-name", "RASPHypervisorBridgeModule",
      "-Xlinker", "-install_name",
      "-Xlinker", "@rpath/${dylib.name}"
    ).inheritIO().start().waitFor()
    check(exit == 0) { "Failed to build RASP hypervisor bridge" }
    hashFile.writeText(swiftSrc.hashCode().toString())
  }

  return Native.load(dylib.absolutePath, MetalBridge::class.java) as MetalBridge
}

/** language=swift */
private val metalBridgeSwiftSource = """
import Foundation
import Metal

private struct Params {
  var vmCount: UInt32
  var workers: UInt32
  var memWords: UInt32
  var outputWords: UInt32
  var quantum: UInt32
  var maxSteps: UInt32
  var vmStride: UInt32
  var memBase: UInt32
}

@_cdecl("metalRunHypervisor")
public func metalRunHypervisor(
  _ msl: UnsafePointer<CChar>?,
  _ kernel: UnsafePointer<CChar>?,
  _ vmCount: Int32,
  _ workers: Int32,
  _ memWords: Int32,
  _ outputWords: Int32,
  _ quantum: Int32,
  _ maxSteps: Int32,
  _ vmWords: UnsafeMutablePointer<Int32>?
) -> Int32 {
  guard
    let msl, let kernel, let vmWords,
    let d = MTLCreateSystemDefaultDevice(),
    let q = d.makeCommandQueue(),
    let lib = try? d.makeLibrary(source: String(cString: msl), options: nil),
    let fn = lib.makeFunction(name: String(cString: kernel)),
    let p = try? d.makeComputePipelineState(function: fn)
  else { return 0 }

  let outStride = Int(outputWords) + 1
  let vmStride = 4 + outStride + Int(memWords)
  let totalWords = Int(vmCount) * vmStride
  let totalBytes = totalWords * MemoryLayout<UInt32>.stride

  guard
    let buf = d.makeBuffer(length: totalBytes, options: .storageModeShared),
    let cb = q.makeCommandBuffer(),
    let ce = cb.makeComputeCommandEncoder()
  else { return 0 }

  memcpy(buf.contents(), vmWords, totalBytes)

  var params = Params(
    vmCount: UInt32(vmCount),
    workers: UInt32(workers),
    memWords: UInt32(memWords),
    outputWords: UInt32(outputWords),
    quantum: UInt32(quantum),
    maxSteps: UInt32(maxSteps),
    vmStride: UInt32(vmStride),
    memBase: UInt32(4 + outStride)
  )

  ce.setComputePipelineState(p)
  ce.setBuffer(buf, offset: 0, index: 0)
  ce.setBytes(&params, length: MemoryLayout<Params>.stride, index: 1)

  let tptg = min(p.threadExecutionWidth, p.maxTotalThreadsPerThreadgroup)
  let numThreadgroups = (Int(workers) + tptg - 1) / tptg

  ce.dispatchThreadgroups(
    MTLSize(width: numThreadgroups, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
  )
  ce.endEncoding()

  cb.commit()
  cb.waitUntilCompleted()
  if cb.error != nil { return 0 }

  memcpy(vmWords, buf.contents(), totalBytes)
  return 1
}"""

/* language=c++ */
internal const val RASP_MSL = """
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint vmCount;
    uint workers;
    uint memWords;
    uint outputWords;
    uint quantum;
    uint maxSteps;
    uint vmStride;
    uint memBase;
};

enum : uint {
    PC = 0u,
    ACC = 1u,
    STEPS = 2u,
    HALTED = 3u,
    OUT_COUNT = 4u,
};

constant uint OPCODE_BITS = 3u;
constant uint OPCODE_MASK = 7u;
constant uint WORD_MASK = 0x1fffffffu;

inline uint vm_off(uint vm, uint off, constant Params& p) { return vm * p.vmStride + off; }
inline uint mem_off(uint vm, uint addr, constant Params& p) { return vm_off(vm, p.memBase + (addr % p.memWords), p); }
inline uint norm_word(uint x) { return x & WORD_MASK; }
inline uint opcode_of(uint ins) { return ins & OPCODE_MASK; }
inline uint operand_of(uint ins) { return ins >> OPCODE_BITS; }
inline uint add_word(uint x, uint y) { return (x + y) & WORD_MASK; }
inline uint mul_word(uint x, uint y) {
    return uint(((ulong)(x & WORD_MASK) * (ulong)(y & WORD_MASK)) & (ulong)WORD_MASK);
}

kernel void run_hypervisor(
    device uint* buf        [[buffer(0)]],
    constant Params& p      [[buffer(1)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid >= p.workers) return;

    for (uint vm = gid; vm < p.vmCount; vm += p.workers) {
        const uint base = vm * p.vmStride;

        uint halted = buf[base + HALTED];
        uint steps  = buf[base + STEPS];
        if (halted != 0u || steps >= p.maxSteps) continue;

        uint pc  = buf[base + PC];
        uint acc = norm_word(buf[base + ACC]);

        while (halted == 0u && steps < p.maxSteps) {
            const uint ins = buf[mem_off(vm, pc, p)];
            const uint op = opcode_of(ins);
            const uint arg = operand_of(ins);

            switch (op) {
                case 1u: // LOD imm
                    acc = norm_word(arg);
                    pc += 1u;
                    break;

                case 2u: // ADD mem[arg]
                    acc = add_word(acc, buf[mem_off(vm, arg, p)]);
                    pc += 1u;
                    break;

                case 3u: // MUL mem[arg]
                    acc = mul_word(acc, buf[mem_off(vm, arg, p)]);
                    pc += 1u;
                    break;

                case 4u: // STO mem[arg] <- acc
                    buf[mem_off(vm, arg, p)] = norm_word(acc);
                    pc += 1u;
                    break;

                case 5u: // BNZ
                    pc = (acc != 0u) ? arg : (pc + 1u);
                    break;

                case 7u: { // PRI mem[arg]
                    uint count = buf[base + OUT_COUNT];
                    if (count < p.outputWords) {
                        buf[base + OUT_COUNT + 1u + count] = buf[mem_off(vm, arg, p)];
                        buf[base + OUT_COUNT] = count + 1u;
                    }
                    pc += 1u;
                    break;
                }

                default: // reserved invalid packed instruction, e.g. 0 => HLT
                    halted = 1u;
                    break;
            }

            steps += 1u;
        }

        if (steps >= p.maxSteps) halted = 1u;

        buf[base + PC] = norm_word(pc);
        buf[base + ACC] = norm_word(acc);
        buf[base + STEPS] = steps;
        buf[base + HALTED] = halted;
    }
}"""
