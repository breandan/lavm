package org.example

import com.sun.jna.Library
import com.sun.jna.Native
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.util.Locale
import java.util.stream.IntStream
import kotlin.math.min
import kotlin.system.measureNanoTime

private const val CJM_HOT_STATE_WORDS = 5
private const val CJM_DEFAULT_SERIALIZED_OUTPUT_WORDS = 3
private const val CJM_DEFAULT_OUTPUT_WORDS = 9
private const val CJM_DEFAULT_MAX_STEPS = 1_000_000
private const val CJM_DEFAULT_WORKERS = 65_536
private const val CJM_OPCODE_BITS = 3
private const val CJM_OPCODE_MASK = (1 shl CJM_OPCODE_BITS) - 1
private const val CJM_WORD_BITS = 29
private const val CJM_WORD_MASK = (1 shl CJM_WORD_BITS) - 1
private const val CJM_TOP_K = 20

/*
./gradlew compareJVMvMetal

Useful overrides:
RASP_VM_FILE=cuda/rasp_vms.bin
RASP_VM_COUNT=100000
RASP_MEM_WORDS=255
RASP_MAX_STEPS=1000000
RASP_WORKERS=65536
RASP_METAL_MODE=compact
RASP_METAL_QUANTUM=128
RASP_METAL_CUTOFF=128
RASP_METAL_SPECIALIZE_LAYOUT=false
RASP_TOP_K=20
*/
fun main(args: Array<String>) {
  val file = File(args.firstOrNull() ?: stringSetting("rasp.file", "RASP_VM_FILE", "cuda/rasp_vms.bin"))
  val serializedOutputWords = intSetting("rasp.serializedOutputWords", "RASP_SERIALIZED_OUTPUT_WORDS", CJM_DEFAULT_SERIALIZED_OUTPUT_WORDS)
  val outputWords = intSetting("rasp.outputWords", "RASP_OUTPUT_WORDS", CJM_DEFAULT_OUTPUT_WORDS)
  val maxSteps = intSetting("rasp.maxSteps", "RASP_MAX_STEPS", CJM_DEFAULT_MAX_STEPS)
  val workers = intSetting("rasp.workers", "RASP_WORKERS", CJM_DEFAULT_WORKERS)
  val metalMode = stringSetting("rasp.metalMode", "RASP_METAL_MODE", "compact").lowercase(Locale.US)
  val metalQuantum = intSetting("rasp.metalQuantum", "RASP_METAL_QUANTUM", 128)
  val metalCutoff = intSetting("rasp.metalCutoff", "RASP_METAL_CUTOFF", 128)
  val specializeMetalLayout = booleanSetting("rasp.metalSpecializeLayout", "RASP_METAL_SPECIALIZE_LAYOUT", false)
  val topK = intSetting("rasp.topK", "RASP_TOP_K", CJM_TOP_K)
  val memWordsOverride = optionalIntSetting("rasp.memWords", "RASP_MEM_WORDS")
  val vmCountLimit = optionalIntSetting("rasp.vmCount", "RASP_VM_COUNT")

  require(file.isFile) { "VM file not found: ${file.absolutePath}" }
  require(serializedOutputWords >= 0) { "Serialized output words must be non-negative" }
  require(outputWords > 0) { "Runtime output words must be positive" }
  require(maxSteps > 0) { "Max steps must be positive" }
  require(workers > 0) { "Worker count must be positive" }
  require(metalMode in setOf("onepass", "queue", "compact", "twophase")) { "Metal mode must be onepass, queue, compact, or twophase" }
  require(metalQuantum > 0) { "Metal quantum must be positive" }
  require(metalCutoff >= 0) { "Metal cutoff must be non-negative" }
  require(topK > 0) { "Top-K must be positive" }

  val layout = detectLayout(file, serializedOutputWords, memWordsOverride)
  val vmCount = min(vmCountLimit ?: layout.records, layout.records)
  require(vmCount > 0) { "VM count must be positive" }

  println("file=${file.path}")
  println("layout: records=${layout.records} memWords=${layout.memWords} serializedOutputWords=$serializedOutputWords runtimeOutputWords=$outputWords")
  println("run: vmCount=$vmCount maxSteps=$maxSteps workers=$workers metalMode=$metalMode metalQuantum=$metalQuantum metalCutoff=$metalCutoff metalSpecializeLayout=$specializeMetalLayout topK=$topK")

  val initial: CompareState
  val realLoadNs = measureNanoTime {
    initial = loadState(file, vmCount, layout.memWords, serializedOutputWords, outputWords)
  }
  println("load: ${formatSeconds(realLoadNs)}s")

  val jvmState = initial.copyDeep()
  val jvmNs = measureNanoTime { runJvm(jvmState, maxSteps) }
  val jvmHeavyHitters = heavyHitters(jvmState, topK)
  println("JVM:   host=${formatSeconds(jvmNs)}s rate=${formatRate(vmCount, jvmNs)} vm/s")
  println("       ${executionSummary(jvmState, maxSteps)}")

  val metalState = initial.copyDeep()
  val metalSource = compareMsl(initial.memWords, outputWords, specializeMetalLayout)
  val bridge = compareMetalBridge()
  warmUpMetal(bridge, metalSource, initial.memWords, outputWords)
  var metalKernelNs = 0L
  var activeAfterCutoff: Int? = null
  val metalHostNs = measureNanoTime {
    if (metalMode == "compact") {
      val kernelNsOut = LongArray(1)
      runMetalKernel(
        bridge = bridge,
        metalSource = metalSource,
        kernelName = "run_hypervisor_simd_compact",
        state = metalState,
        vmCount = vmCount,
        workers = min(workers, vmCount),
        maxSteps = maxSteps,
        quantum = metalCutoff.coerceIn(1, maxSteps),
        markTimeout = true,
        kernelNsOut = kernelNsOut
      )
      metalKernelNs += kernelNsOut[0]
    } else if (metalMode == "queue") {
      val kernelNsOut = LongArray(1)
      runMetalKernel(
        bridge = bridge,
        metalSource = metalSource,
        kernelName = "run_hypervisor_queue",
        state = metalState,
        vmCount = vmCount,
        workers = min(workers, vmCount),
        maxSteps = maxSteps,
        quantum = metalQuantum,
        markTimeout = true,
        kernelNsOut = kernelNsOut
      )
      metalKernelNs += kernelNsOut[0]
    } else if (metalMode == "twophase" && metalCutoff in 1 until maxSteps) {
      val cutoffKernelNs = LongArray(1)
      runMetalKernel(
        bridge = bridge,
        metalSource = metalSource,
        kernelName = "run_hypervisor",
        state = metalState,
        vmCount = vmCount,
        workers = min(workers, vmCount),
        maxSteps = metalCutoff,
        quantum = metalQuantum,
        markTimeout = false,
        kernelNsOut = cutoffKernelNs
      )
      metalKernelNs += cutoffKernelNs[0]

      val activeVms = activeVmIndices(metalState, maxSteps)
      activeAfterCutoff = activeVms.size
      if (activeVms.isNotEmpty()) {
        val compactState = compactState(metalState, activeVms)
        val compactKernelNs = LongArray(1)
        runMetalKernel(
          bridge = bridge,
          metalSource = metalSource,
          kernelName = "run_hypervisor",
          state = compactState,
          vmCount = compactState.vmCount,
          workers = min(workers, compactState.vmCount),
          maxSteps = maxSteps,
          quantum = metalQuantum,
          markTimeout = true,
          kernelNsOut = compactKernelNs
        )
        metalKernelNs += compactKernelNs[0]
        scatterState(compactState, activeVms, metalState)
      }
    } else {
      val kernelNsOut = LongArray(1)
      runMetalKernel(
        bridge = bridge,
        metalSource = metalSource,
        kernelName = "run_hypervisor",
        state = metalState,
        vmCount = vmCount,
        workers = min(workers, vmCount),
        maxSteps = maxSteps,
        quantum = metalQuantum,
        markTimeout = true,
        kernelNsOut = kernelNsOut
      )
      metalKernelNs += kernelNsOut[0]
    }
  }
  val metalHeavyHitters = heavyHitters(metalState, topK)
  val activeSuffix = activeAfterCutoff?.let { " activeAfterCutoff=$it" } ?: ""
  println("Metal: host=${formatSeconds(metalHostNs)}s kernel=${formatSeconds(metalKernelNs)}s hostRate=${formatRate(vmCount, metalHostNs)} vm/s$activeSuffix")
  println("       ${executionSummary(metalState, maxSteps)}")

  val stateMismatch = firstStateMismatch(jvmState, metalState)
  require(stateMismatch == null) { stateMismatch ?: "State mismatch" }

  if (jvmHeavyHitters != metalHeavyHitters) {
    println("JVM heavy hitters:")
    printHeavyHitters(jvmHeavyHitters)
    println("Metal heavy hitters:")
    printHeavyHitters(metalHeavyHitters)
    error("Heavy hitters differ")
  }

  println("heavy hitters verified identical")
  printHeavyHitters(jvmHeavyHitters)
}

private fun stringSetting(propertyName: String, envName: String, default: String): String =
  System.getProperty(propertyName) ?: System.getenv(envName) ?: default

private fun intSetting(propertyName: String, envName: String, default: Int): Int =
  optionalIntSetting(propertyName, envName) ?: default

private fun optionalIntSetting(propertyName: String, envName: String): Int? =
  System.getProperty(propertyName)?.toIntOrNull() ?: System.getenv(envName)?.toIntOrNull()

private fun booleanSetting(propertyName: String, envName: String, default: Boolean): Boolean =
  System.getProperty(propertyName)?.toBooleanStrictOrNull()
    ?: System.getenv(envName)?.toBooleanStrictOrNull()
    ?: default

private data class CompareLayout(val records: Int, val memWords: Int)

private fun detectLayout(file: File, serializedOutputWords: Int, memWordsOverride: Int?): CompareLayout {
  require(file.length() % Int.SIZE_BYTES == 0L) {
    "File length must be a multiple of ${Int.SIZE_BYTES}: ${file.length()}"
  }

  val totalInts = file.length() / Int.SIZE_BYTES
  val candidates = if (memWordsOverride != null) {
    listOf(memWordsOverride)
  } else {
    listOf(128, 255, 256)
  }

  val matches = candidates.mapNotNull { memWords ->
    require(memWords > 0) { "Memory words must be positive" }
    val intsPerVm = CJM_HOT_STATE_WORDS + serializedOutputWords + memWords + 1
    if (totalInts % intsPerVm == 0L) CompareLayout((totalInts / intsPerVm).toInt(), memWords) else null
  }

  require(matches.isNotEmpty()) { "Could not detect VM layout for ${file.absolutePath}; pass RASP_MEM_WORDS or -Drasp.memWords" }
  require(matches.size == 1) { "Ambiguous VM layout candidates: $matches; pass RASP_MEM_WORDS or -Drasp.memWords" }

  return matches.single()
}

private data class CompareState(
  val vmCount: Int,
  val memWords: Int,
  val outputWords: Int,
  val pcs: IntArray,
  val accs: IntArray,
  val stepss: IntArray,
  val halteds: IntArray,
  val outCounts: IntArray,
  val outputs: IntArray,
  val mems: IntArray,
) {
  fun copyDeep(): CompareState =
    copy(
      pcs = pcs.copyOf(),
      accs = accs.copyOf(),
      stepss = stepss.copyOf(),
      halteds = halteds.copyOf(),
      outCounts = outCounts.copyOf(),
      outputs = outputs.copyOf(),
      mems = mems.copyOf()
    )
}

private fun loadState(
  file: File,
  vmCount: Int,
  memWords: Int,
  serializedOutputWords: Int,
  outputWords: Int,
): CompareState {
  val state = CompareState(
    vmCount = vmCount,
    memWords = memWords,
    outputWords = outputWords,
    pcs = IntArray(vmCount),
    accs = IntArray(vmCount),
    stepss = IntArray(vmCount),
    halteds = IntArray(vmCount),
    outCounts = IntArray(vmCount),
    outputs = IntArray(vmCount * outputWords),
    mems = IntArray(vmCount * memWords)
  )

  DataInputStream(BufferedInputStream(FileInputStream(file))).use { input ->
    var vm = 0
    while (vm < vmCount) {
      state.pcs[vm] = input.readInt()
      state.accs[vm] = input.readInt()
      state.stepss[vm] = input.readInt()
      state.halteds[vm] = input.readInt()
      state.outCounts[vm] = input.readInt()

      var out = 0
      val outBase = vm * outputWords
      while (out < serializedOutputWords) {
        val value = input.readInt()
        if (out < outputWords) state.outputs[outBase + out] = value
        out++
      }

      var mem = 0
      val memBase = vm * memWords
      while (mem < memWords) {
        state.mems[memBase + mem] = input.readInt()
        mem++
      }

      val terminator = input.readInt()
      require(terminator == Int.MAX_VALUE) {
        "VM $vm: expected terminator ${Int.MAX_VALUE}, got $terminator"
      }

      vm++
    }
  }

  return state
}

private fun runJvm(state: CompareState, maxSteps: Int) {
  IntStream.range(0, state.vmCount).parallel().forEach { vm ->
    var pc = state.pcs[vm]
    var acc = state.accs[vm] and CJM_WORD_MASK
    var steps = state.stepss[vm]
    var halted = state.halteds[vm]
    var outCount = state.outCounts[vm]

    val memBase = vm * state.memWords
    val outBase = vm * state.outputWords

    while (halted == 0 && steps < maxSteps) {
      val ins = state.mems[memBase + wrap(pc, state.memWords)]
      val op = ins and CJM_OPCODE_MASK
      val arg = ins ushr CJM_OPCODE_BITS

      when (op) {
        1 -> {
          acc = normWord(arg)
          pc += 1
        }

        2 -> {
          acc = addWord(acc, state.mems[memBase + wrap(arg, state.memWords)])
          pc += 1
        }

        3 -> {
          acc = mulWord(acc, state.mems[memBase + wrap(arg, state.memWords)])
          pc += 1
        }

        4 -> {
          state.mems[memBase + wrap(arg, state.memWords)] = normWord(acc)
          pc += 1
        }

        5 -> {
          pc = if (acc != 0) arg else pc + 1
        }

        7 -> {
          if (outCount < state.outputWords) {
            state.outputs[outBase + outCount] = state.mems[memBase + wrap(arg, state.memWords)]
            outCount += 1
          }
          pc += 1
        }

        else -> {
          halted = 1
        }
      }

      steps += 1
    }

    if (steps >= maxSteps) halted = 1

    state.pcs[vm] = normWord(pc)
    state.accs[vm] = normWord(acc)
    state.stepss[vm] = steps
    state.halteds[vm] = halted
    state.outCounts[vm] = outCount
  }
}

private fun runMetalKernel(
  bridge: CompareMetalBridge,
  metalSource: String,
  kernelName: String,
  state: CompareState,
  vmCount: Int,
  workers: Int,
  maxSteps: Int,
  quantum: Int,
  markTimeout: Boolean,
  kernelNsOut: LongArray,
) {
  val ok = bridge.compareRunHypervisor(
    metalSource,
    kernelName,
    vmCount,
    workers,
    state.memWords,
    state.outputWords,
    maxSteps,
    quantum,
    if (markTimeout) 1 else 0,
    state.pcs,
    state.accs,
    state.stepss,
    state.halteds,
    state.outCounts,
    state.outputs,
    state.mems,
    kernelNsOut
  )
  require(ok != 0) { "compareRunHypervisor failed" }
}

private fun activeVmIndices(state: CompareState, finalMaxSteps: Int): IntArray {
  val active = IntArray(state.vmCount)
  var activeCount = 0
  var vm = 0
  while (vm < state.vmCount) {
    if (state.halteds[vm] == 0 && state.stepss[vm] < finalMaxSteps) {
      active[activeCount] = vm
      activeCount++
    }
    vm++
  }
  return active.copyOf(activeCount)
}

private fun compactState(source: CompareState, activeVms: IntArray): CompareState {
  val compact = CompareState(
    vmCount = activeVms.size,
    memWords = source.memWords,
    outputWords = source.outputWords,
    pcs = IntArray(activeVms.size),
    accs = IntArray(activeVms.size),
    stepss = IntArray(activeVms.size),
    halteds = IntArray(activeVms.size),
    outCounts = IntArray(activeVms.size),
    outputs = IntArray(activeVms.size * source.outputWords),
    mems = IntArray(activeVms.size * source.memWords)
  )

  var compactVm = 0
  while (compactVm < activeVms.size) {
    val sourceVm = activeVms[compactVm]
    compact.pcs[compactVm] = source.pcs[sourceVm]
    compact.accs[compactVm] = source.accs[sourceVm]
    compact.stepss[compactVm] = source.stepss[sourceVm]
    compact.halteds[compactVm] = source.halteds[sourceVm]
    compact.outCounts[compactVm] = source.outCounts[sourceVm]

    System.arraycopy(
      source.outputs,
      sourceVm * source.outputWords,
      compact.outputs,
      compactVm * source.outputWords,
      source.outputWords
    )
    System.arraycopy(
      source.mems,
      sourceVm * source.memWords,
      compact.mems,
      compactVm * source.memWords,
      source.memWords
    )

    compactVm++
  }

  return compact
}

private fun scatterState(compact: CompareState, activeVms: IntArray, target: CompareState) {
  var compactVm = 0
  while (compactVm < activeVms.size) {
    val targetVm = activeVms[compactVm]
    target.pcs[targetVm] = compact.pcs[compactVm]
    target.accs[targetVm] = compact.accs[compactVm]
    target.stepss[targetVm] = compact.stepss[compactVm]
    target.halteds[targetVm] = compact.halteds[compactVm]
    target.outCounts[targetVm] = compact.outCounts[compactVm]

    System.arraycopy(
      compact.outputs,
      compactVm * compact.outputWords,
      target.outputs,
      targetVm * target.outputWords,
      target.outputWords
    )
    System.arraycopy(
      compact.mems,
      compactVm * compact.memWords,
      target.mems,
      targetVm * target.memWords,
      target.memWords
    )

    compactVm++
  }
}

private fun wrap(x: Int, n: Int): Int = Integer.remainderUnsigned(x, n)

private fun normWord(x: Int): Int = x and CJM_WORD_MASK

private fun addWord(x: Int, y: Int): Int = (x + y) and CJM_WORD_MASK

private fun mulWord(x: Int, y: Int): Int =
  (((x and CJM_WORD_MASK).toLong() * (y and CJM_WORD_MASK).toLong()) and CJM_WORD_MASK.toLong()).toInt()

private data class HeavyHitter(
  val vm: Int,
  val steps: Int,
  val halted: Int,
  val pc: Int,
  val acc: Int,
  val outCount: Int,
)

private fun heavyHitters(state: CompareState, topK: Int): List<HeavyHitter> =
  (0 until state.vmCount)
    .asSequence()
    .map { vm ->
      HeavyHitter(
        vm = vm,
        steps = state.stepss[vm],
        halted = state.halteds[vm],
        pc = state.pcs[vm],
        acc = state.accs[vm],
        outCount = state.outCounts[vm]
      )
    }
    .sortedWith(compareByDescending<HeavyHitter> { it.steps }.thenBy { it.vm })
    .take(topK)
    .toList()

private fun printHeavyHitters(hitters: List<HeavyHitter>) {
  hitters.forEachIndexed { index, hitter ->
    println(
      "  ${index + 1}. vm=${hitter.vm} steps=${hitter.steps} halted=${hitter.halted} " +
        "pc=${hitter.pc} acc=${hitter.acc} outCount=${hitter.outCount}"
    )
  }
}

private data class ExecutionSummary(
  val totalSteps: Long,
  val naturalHalts: Int,
  val timeouts: Int,
  val nonHalted: Int,
) {
  override fun toString(): String =
    "totalSteps=$totalSteps naturalHalts=$naturalHalts timeouts=$timeouts nonHalted=$nonHalted"
}

private fun executionSummary(state: CompareState, maxSteps: Int): ExecutionSummary {
  var totalSteps = 0L
  var naturalHalts = 0
  var timeouts = 0
  var nonHalted = 0

  var vm = 0
  while (vm < state.vmCount) {
    val steps = state.stepss[vm]
    val halted = state.halteds[vm] != 0
    totalSteps += steps.toLong()

    when {
      halted && steps >= maxSteps -> timeouts++
      halted -> naturalHalts++
      else -> nonHalted++
    }
    vm++
  }

  return ExecutionSummary(totalSteps, naturalHalts, timeouts, nonHalted)
}

private fun firstStateMismatch(expected: CompareState, actual: CompareState): String? {
  require(expected.vmCount == actual.vmCount)
  require(expected.memWords == actual.memWords)
  require(expected.outputWords == actual.outputWords)

  firstMismatch("pc", expected.pcs, actual.pcs)?.let { return it }
  firstMismatch("acc", expected.accs, actual.accs)?.let { return it }
  firstMismatch("steps", expected.stepss, actual.stepss)?.let { return it }
  firstMismatch("halted", expected.halteds, actual.halteds)?.let { return it }
  firstMismatch("outCount", expected.outCounts, actual.outCounts)?.let { return it }
  firstMismatch("outputs", expected.outputs, actual.outputs)?.let { return it }
  firstMismatch("mem", expected.mems, actual.mems)?.let { return it }
  return null
}

private fun firstMismatch(name: String, expected: IntArray, actual: IntArray): String? {
  var i = 0
  while (i < expected.size) {
    if (expected[i] != actual[i]) return "$name mismatch at index $i: JVM=${expected[i]} Metal=${actual[i]}"
    i++
  }
  return null
}

private fun warmUpMetal(bridge: CompareMetalBridge, metalSource: String, memWords: Int, outputWords: Int) {
  val kernelNs = LongArray(1)
  val ok = bridge.compareRunHypervisor(
    metalSource,
    "run_hypervisor",
    1,
    1,
    memWords,
    outputWords,
    1,
    1,
    1,
    IntArray(1),
    IntArray(1),
    IntArray(1),
    IntArray(1),
    IntArray(1),
    IntArray(outputWords),
    IntArray(memWords),
    kernelNs
  )
  require(ok != 0) { "Metal warm-up failed" }
}

interface CompareMetalBridge : Library {
  fun compareRunHypervisor(
    mslSource: String,
    kernelName: String,
    vmCount: Int,
    workers: Int,
    memWords: Int,
    outputWords: Int,
    maxSteps: Int,
    quantum: Int,
    markTimeout: Int,
    pcs: IntArray,
    accs: IntArray,
    stepss: IntArray,
    halteds: IntArray,
    outCounts: IntArray,
    outputs: IntArray,
    mems: IntArray,
    kernelElapsedNs: LongArray,
  ): Int
}

private fun compareMetalBridge(): CompareMetalBridge {
  if (!System.getProperty("os.name").startsWith("Mac")) error("Apple Metal comparison requires macOS")
  val directory = File("build/compareJvmMetal").also { it.mkdirs() }
  val swiftFile = directory.resolve("CompareRASPBridge.swift")
  val dylib = directory.resolve("libCompareRASPBridge.dylib")
  val hashFile = directory.resolve(".compareRaspBridgeHash")
  val swiftSrc = CJM_COMPARE_SWIFT_SOURCE

  if (!dylib.exists() || !hashFile.exists() || hashFile.readText() != swiftSrc.hashCode().toString()) {
    swiftFile.writeText(swiftSrc)
    val exit = ProcessBuilder(
      "xcrun", "swiftc", "-O", "-emit-library",
      swiftFile.absolutePath,
      "-o", dylib.absolutePath,
      "-module-name", "CompareRASPBridgeModule",
      "-Xlinker", "-install_name",
      "-Xlinker", "@rpath/${dylib.name}"
    ).inheritIO().start().waitFor()
    check(exit == 0) { "Failed to build CompareRASP bridge" }
    hashFile.writeText(swiftSrc.hashCode().toString())
  }

  return Native.load(dylib.absolutePath, CompareMetalBridge::class.java) as CompareMetalBridge
}

private fun formatSeconds(ns: Long): String = "%.6f".format(Locale.US, ns / 1e9)

private fun formatRate(vmCount: Int, ns: Long): String {
  val secs = ns / 1e9
  val rate = if (secs > 0.0) vmCount / secs else Double.POSITIVE_INFINITY
  return "%.1f".format(Locale.US, rate)
}

/** language=swift */
private val CJM_COMPARE_SWIFT_SOURCE = """
import Foundation
import Metal

private struct Params {
  var vmCount: UInt32
  var workers: UInt32
  var memWords: UInt32
  var outputWords: UInt32
  var maxSteps: UInt32
  var quantum: UInt32
  var markTimeout: UInt32
}

private final class BridgeState {
  static let shared = BridgeState()

  let device: MTLDevice
  let queue: MTLCommandQueue
  private var pipelines: [String: MTLComputePipelineState] = [:]

  private init?() {
    guard
      let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue()
    else { return nil }

    self.device = device
    self.queue = queue
  }

  func pipeline(msl: String, kernel: String) throws -> MTLComputePipelineState {
    let key = "\(kernel):\(msl.hashValue)"
    if let cached = pipelines[key] { return cached }

    let lib = try device.makeLibrary(source: msl, options: nil)
    guard let fn = lib.makeFunction(name: kernel) else {
      throw NSError(domain: "CompareRASPBridge", code: 1)
    }
    let pipeline = try device.makeComputePipelineState(function: fn)
    pipelines[key] = pipeline
    return pipeline
  }
}

private func makeBuffer(_ device: MTLDevice, _ ptr: UnsafeMutableRawPointer, _ words: Int) -> MTLBuffer? {
  let bytes = words * MemoryLayout<Int32>.stride
  guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else { return nil }
  memcpy(buffer.contents(), ptr, bytes)
  return buffer
}

private func copyBack(_ buffer: MTLBuffer, _ ptr: UnsafeMutableRawPointer, _ words: Int) {
  memcpy(ptr, buffer.contents(), words * MemoryLayout<Int32>.stride)
}

@_cdecl("compareRunHypervisor")
public func compareRunHypervisor(
  _ msl: UnsafePointer<CChar>?,
  _ kernel: UnsafePointer<CChar>?,
  _ vmCount: Int32,
  _ workers: Int32,
  _ memWords: Int32,
  _ outputWords: Int32,
  _ maxSteps: Int32,
  _ quantum: Int32,
  _ markTimeout: Int32,
  _ pcs: UnsafeMutablePointer<Int32>?,
  _ accs: UnsafeMutablePointer<Int32>?,
  _ stepss: UnsafeMutablePointer<Int32>?,
  _ halteds: UnsafeMutablePointer<Int32>?,
  _ outCounts: UnsafeMutablePointer<Int32>?,
  _ outputs: UnsafeMutablePointer<Int32>?,
  _ mems: UnsafeMutablePointer<Int32>?,
  _ kernelElapsedNs: UnsafeMutablePointer<Int64>?
) -> Int32 {
  guard
    let msl, let kernel,
    let pcs, let accs, let stepss, let halteds, let outCounts, let outputs, let mems,
    let state = BridgeState.shared
  else { return 0 }

  do {
    let mslString = String(cString: msl)
    let kernelName = String(cString: kernel)
    let vmCountInt = Int(vmCount)
    let memWordsInt = Int(memWords)
    let outputWordsInt = Int(outputWords)
    let memTotalWords = vmCountInt * memWordsInt
    let outputTotalWords = vmCountInt * outputWordsInt
    var nextVm: UInt32 = 0
    var activeCount: UInt32 = 0

    guard
      let pcsBuffer = makeBuffer(state.device, pcs, vmCountInt),
      let accsBuffer = makeBuffer(state.device, accs, vmCountInt),
      let stepssBuffer = makeBuffer(state.device, stepss, vmCountInt),
      let haltedsBuffer = makeBuffer(state.device, halteds, vmCountInt),
      let outCountsBuffer = makeBuffer(state.device, outCounts, vmCountInt),
      let outputsBuffer = makeBuffer(state.device, outputs, outputTotalWords),
      let memsBuffer = makeBuffer(state.device, mems, memTotalWords),
      let nextVmBuffer = state.device.makeBuffer(bytes: &nextVm, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
      let activeIdsBuffer = state.device.makeBuffer(length: vmCountInt * MemoryLayout<UInt32>.stride, options: .storageModeShared),
      let activeCountBuffer = state.device.makeBuffer(bytes: &activeCount, length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
      let commandBuffer = state.queue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else { return 0 }

    var params = Params(
      vmCount: UInt32(vmCount),
      workers: UInt32(workers),
      memWords: UInt32(memWords),
      outputWords: UInt32(outputWords),
      maxSteps: UInt32(maxSteps),
      quantum: UInt32(quantum),
      markTimeout: UInt32(markTimeout)
    )

    if kernelName == "run_hypervisor_simd_compact" {
      let cutoffPipeline = try state.pipeline(msl: mslString, kernel: "run_hypervisor_cutoff_collect")

      encoder.setComputePipelineState(cutoffPipeline)
      encoder.setBuffer(pcsBuffer, offset: 0, index: 0)
      encoder.setBuffer(accsBuffer, offset: 0, index: 1)
      encoder.setBuffer(stepssBuffer, offset: 0, index: 2)
      encoder.setBuffer(haltedsBuffer, offset: 0, index: 3)
      encoder.setBuffer(outCountsBuffer, offset: 0, index: 4)
      encoder.setBuffer(outputsBuffer, offset: 0, index: 5)
      encoder.setBuffer(memsBuffer, offset: 0, index: 6)
      encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
      encoder.setBuffer(activeIdsBuffer, offset: 0, index: 8)
      encoder.setBuffer(activeCountBuffer, offset: 0, index: 9)

      let cutoffThreadsPerGroup = min(cutoffPipeline.threadExecutionWidth, cutoffPipeline.maxTotalThreadsPerThreadgroup)
      let cutoffThreadgroups = (Int(workers) + cutoffThreadsPerGroup - 1) / cutoffThreadsPerGroup
      encoder.dispatchThreadgroups(
        MTLSize(width: cutoffThreadgroups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: cutoffThreadsPerGroup, height: 1, depth: 1)
      )
      encoder.endEncoding()

      let start = DispatchTime.now().uptimeNanoseconds
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      if commandBuffer.error != nil { return 0 }

      let activeCountValue = activeCountBuffer.contents().load(as: UInt32.self)
      if activeCountValue > 0 {
        var activeParams = params
        activeParams.vmCount = activeCountValue
        guard
          let activeCommandBuffer = state.queue.makeCommandBuffer(),
          let activeEncoder = activeCommandBuffer.makeComputeCommandEncoder()
        else { return 0 }

        let activePipeline = try state.pipeline(msl: mslString, kernel: "run_hypervisor_active")
        let activeWorkers = min(Int(workers), Int(activeCountValue))
        activeParams.workers = UInt32(activeWorkers)

        activeEncoder.setComputePipelineState(activePipeline)
        activeEncoder.setBuffer(pcsBuffer, offset: 0, index: 0)
        activeEncoder.setBuffer(accsBuffer, offset: 0, index: 1)
        activeEncoder.setBuffer(stepssBuffer, offset: 0, index: 2)
        activeEncoder.setBuffer(haltedsBuffer, offset: 0, index: 3)
        activeEncoder.setBuffer(outCountsBuffer, offset: 0, index: 4)
        activeEncoder.setBuffer(outputsBuffer, offset: 0, index: 5)
        activeEncoder.setBuffer(memsBuffer, offset: 0, index: 6)
        activeEncoder.setBytes(&activeParams, length: MemoryLayout<Params>.stride, index: 7)
        activeEncoder.setBuffer(activeIdsBuffer, offset: 0, index: 8)

        let activeThreadsPerGroup = min(activePipeline.threadExecutionWidth, activePipeline.maxTotalThreadsPerThreadgroup)
        let activeThreadgroups = (activeWorkers + activeThreadsPerGroup - 1) / activeThreadsPerGroup
        activeEncoder.dispatchThreadgroups(
          MTLSize(width: activeThreadgroups, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: activeThreadsPerGroup, height: 1, depth: 1)
        )
        activeEncoder.endEncoding()
        activeCommandBuffer.commit()
        activeCommandBuffer.waitUntilCompleted()
        if activeCommandBuffer.error != nil { return 0 }
      }

      let end = DispatchTime.now().uptimeNanoseconds
      kernelElapsedNs?.pointee = Int64(end - start)

      copyBack(pcsBuffer, pcs, vmCountInt)
      copyBack(accsBuffer, accs, vmCountInt)
      copyBack(stepssBuffer, stepss, vmCountInt)
      copyBack(haltedsBuffer, halteds, vmCountInt)
      copyBack(outCountsBuffer, outCounts, vmCountInt)
      copyBack(outputsBuffer, outputs, outputTotalWords)
      copyBack(memsBuffer, mems, memTotalWords)
      return 1
    }

    let pipeline = try state.pipeline(msl: mslString, kernel: kernelName)

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(pcsBuffer, offset: 0, index: 0)
    encoder.setBuffer(accsBuffer, offset: 0, index: 1)
    encoder.setBuffer(stepssBuffer, offset: 0, index: 2)
    encoder.setBuffer(haltedsBuffer, offset: 0, index: 3)
    encoder.setBuffer(outCountsBuffer, offset: 0, index: 4)
    encoder.setBuffer(outputsBuffer, offset: 0, index: 5)
    encoder.setBuffer(memsBuffer, offset: 0, index: 6)
    encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 7)
    encoder.setBuffer(nextVmBuffer, offset: 0, index: 8)

    let threadsPerGroup = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
    let threadgroups = (Int(workers) + threadsPerGroup - 1) / threadsPerGroup
    encoder.dispatchThreadgroups(
      MTLSize(width: threadgroups, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
    )
    encoder.endEncoding()

    let start = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds
    if commandBuffer.error != nil { return 0 }
    kernelElapsedNs?.pointee = Int64(end - start)

    copyBack(pcsBuffer, pcs, vmCountInt)
    copyBack(accsBuffer, accs, vmCountInt)
    copyBack(stepssBuffer, stepss, vmCountInt)
    copyBack(haltedsBuffer, halteds, vmCountInt)
    copyBack(outCountsBuffer, outCounts, vmCountInt)
    copyBack(outputsBuffer, outputs, outputTotalWords)
    copyBack(memsBuffer, mems, memTotalWords)
    return 1
  } catch {
    return 0
  }
}
"""

private fun compareMsl(memWords: Int, outputWords: Int, specializeLayout: Boolean): String {
  require(memWords > 0) { "Memory words must be positive" }
  require(outputWords > 0) { "Output words must be positive" }

  val layoutConstants = if (specializeLayout) {
    """
constant uint MEM_WORDS = ${memWords}u;
constant uint OUTPUT_WORDS = ${outputWords}u;
""".trim()
  } else {
    ""
  }
  val memWordsExpr = if (specializeLayout) "MEM_WORDS" else "p.memWords"
  val outputWordsExpr = if (specializeLayout) "OUTPUT_WORDS" else "p.outputWords"
  val wrapFunction = metalWrapFunction(memWords, specializeLayout)
  val wrapPc = if (specializeLayout) "wrap(pc)" else "wrap(pc, p)"
  val wrapArg = if (specializeLayout) "wrap(arg)" else "wrap(arg, p)"

  /* language=c++ */
  return """
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint vmCount;
    uint workers;
    uint memWords;
    uint outputWords;
    uint maxSteps;
    uint quantum;
    uint markTimeout;
};

constant uint OPCODE_BITS = 3u;
constant uint OPCODE_MASK = 7u;
constant uint WORD_MASK = 0x1fffffffu;
$layoutConstants

$wrapFunction
inline uint norm_word(uint x) { return x & WORD_MASK; }
inline uint opcode_of(uint ins) { return ins & OPCODE_MASK; }
inline uint operand_of(uint ins) { return ins >> OPCODE_BITS; }
inline uint add_word(uint x, uint y) { return (x + y) & WORD_MASK; }
inline uint mul_word(uint x, uint y) {
    return uint(((ulong)(x & WORD_MASK) * (ulong)(y & WORD_MASK)) & (ulong)WORD_MASK);
}

inline void run_vm_to(
    uint vm,
    uint limit,
    device uint* pcs,
    device uint* accs,
    device uint* stepss,
    device uint* halteds,
    device uint* out_counts,
    device uint* outputs,
    device uint* mems,
    constant Params& p
) {
    uint pc = pcs[vm];
    uint acc = norm_word(accs[vm]);
    uint steps = stepss[vm];
    uint halted = halteds[vm];
    uint out_count = out_counts[vm];

    const uint mem_base = vm * $memWordsExpr;
    const uint out_base = vm * $outputWordsExpr;

    while (halted == 0u && steps < limit) {
        const uint ins = mems[mem_base + $wrapPc];
        const uint op = opcode_of(ins);
        const uint arg = operand_of(ins);

        switch (op) {
            case 1u:
                acc = norm_word(arg);
                pc += 1u;
                break;

            case 2u:
                acc = add_word(acc, mems[mem_base + $wrapArg]);
                pc += 1u;
                break;

            case 3u:
                acc = mul_word(acc, mems[mem_base + $wrapArg]);
                pc += 1u;
                break;

            case 4u:
                mems[mem_base + $wrapArg] = norm_word(acc);
                pc += 1u;
                break;

            case 5u:
                pc = (acc != 0u) ? arg : (pc + 1u);
                break;

            case 7u: {
                const uint value = mems[mem_base + $wrapArg];
                if (out_count < $outputWordsExpr) {
                    outputs[out_base + out_count] = value;
                    ++out_count;
                }
                pc += 1u;
                break;
            }

            default:
                halted = 1u;
                break;
        }

        ++steps;
    }

    if (steps >= p.maxSteps && p.markTimeout != 0u) halted = 1u;

    pcs[vm] = norm_word(pc);
    accs[vm] = norm_word(acc);
    stepss[vm] = steps;
    halteds[vm] = halted;
    out_counts[vm] = out_count;
}

kernel void run_hypervisor(
    device uint* pcs        [[buffer(0)]],
    device uint* accs       [[buffer(1)]],
    device uint* stepss     [[buffer(2)]],
    device uint* halteds    [[buffer(3)]],
    device uint* out_counts [[buffer(4)]],
    device uint* outputs    [[buffer(5)]],
    device uint* mems       [[buffer(6)]],
    constant Params& p      [[buffer(7)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid >= p.workers) return;
    for (uint vm = gid; vm < p.vmCount; vm += p.workers) {
        run_vm_to(vm, p.maxSteps, pcs, accs, stepss, halteds, out_counts, outputs, mems, p);
    }
}

kernel void run_hypervisor_queue(
    device uint* pcs        [[buffer(0)]],
    device uint* accs       [[buffer(1)]],
    device uint* stepss     [[buffer(2)]],
    device uint* halteds    [[buffer(3)]],
    device uint* out_counts [[buffer(4)]],
    device uint* outputs    [[buffer(5)]],
    device uint* mems       [[buffer(6)]],
    constant Params& p      [[buffer(7)]],
    device atomic_uint* next_vm [[buffer(8)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid >= p.workers) return;

    uint vm = atomic_fetch_add_explicit(next_vm, 1u, memory_order_relaxed);
    while (vm < p.vmCount) {
        while (halteds[vm] == 0u && stepss[vm] < p.maxSteps) {
            const uint limit = min(p.maxSteps, stepss[vm] + p.quantum);
            run_vm_to(vm, limit, pcs, accs, stepss, halteds, out_counts, outputs, mems, p);
        }
        vm = atomic_fetch_add_explicit(next_vm, 1u, memory_order_relaxed);
    }
}

kernel void run_hypervisor_cutoff_collect(
    device uint* pcs        [[buffer(0)]],
    device uint* accs       [[buffer(1)]],
    device uint* stepss     [[buffer(2)]],
    device uint* halteds    [[buffer(3)]],
    device uint* out_counts [[buffer(4)]],
    device uint* outputs    [[buffer(5)]],
    device uint* mems       [[buffer(6)]],
    constant Params& p      [[buffer(7)]],
    device uint* active_ids [[buffer(8)]],
    device atomic_uint* active_count [[buffer(9)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid >= p.workers) return;
    const uint cutoff = min(p.maxSteps, p.quantum);

    for (uint vm = gid; vm < p.vmCount; vm += p.workers) {
        run_vm_to(vm, cutoff, pcs, accs, stepss, halteds, out_counts, outputs, mems, p);
        if (halteds[vm] == 0u && stepss[vm] < p.maxSteps) {
            const uint slot = atomic_fetch_add_explicit(active_count, 1u, memory_order_relaxed);
            active_ids[slot] = vm;
        }
    }
}

kernel void run_hypervisor_active(
    device uint* pcs        [[buffer(0)]],
    device uint* accs       [[buffer(1)]],
    device uint* stepss     [[buffer(2)]],
    device uint* halteds    [[buffer(3)]],
    device uint* out_counts [[buffer(4)]],
    device uint* outputs    [[buffer(5)]],
    device uint* mems       [[buffer(6)]],
    constant Params& p      [[buffer(7)]],
    device uint* active_ids [[buffer(8)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid >= p.workers) return;
    for (uint slot = gid; slot < p.vmCount; slot += p.workers) {
        run_vm_to(active_ids[slot], p.maxSteps, pcs, accs, stepss, halteds, out_counts, outputs, mems, p);
    }
}
"""
}

private fun metalWrapFunction(memWords: Int, specializeLayout: Boolean): String =
  when {
    !specializeLayout ->
      "inline uint wrap(uint x, constant Params& p) { return x % p.memWords; }"

    memWords > 0 && memWords and (memWords - 1) == 0 ->
      "inline uint wrap(uint x) { return x & ${memWords - 1}u; }"

    else ->
      "inline uint wrap(uint x) { return x % MEM_WORDS; }"
  }
