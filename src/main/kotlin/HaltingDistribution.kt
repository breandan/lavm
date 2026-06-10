package org.example

import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.util.Locale
import kotlin.math.min
import kotlin.system.measureNanoTime

private const val DEFAULT_CHUNK_VMS = 500_000

/*
./gradlew haltingDistribution

Useful overrides:
RASP_VM_FILE=cuda/rasp_vms_brk.bin
RASP_VM_COUNT=8000000
RASP_CHUNK_VMS=500000
RASP_WORKERS=65536
RASP_QUANTUM=1000
RASP_MAX_STEPS=1000000
*/
fun main() {
  val file = File(stringSetting("rasp.file", "RASP_VM_FILE", DEFAULT_VM_INITIAL_STATES_FILE.path))
  val requestedVmCount = optionalIntSetting("rasp.vmCount", "RASP_VM_COUNT")
  val chunkVms = intSetting("rasp.chunkVms", "RASP_CHUNK_VMS", DEFAULT_CHUNK_VMS)
  val workers = intSetting("rasp.workers", "RASP_WORKERS", MAX_CONCURRENT_THREADS)
  val quantum = intSetting("rasp.quantum", "RASP_QUANTUM", QUANTUM)
  val maxSteps = intSetting("rasp.maxSteps", "RASP_MAX_STEPS", MAX_STEPS)

  require(file.isFile) { "VM file not found: ${file.absolutePath}" }
  require(chunkVms > 0) { "Chunk VM count must be positive" }
  require(workers > 0) { "Worker count must be positive" }
  require(quantum > 0) { "Quantum must be positive" }
  require(maxSteps > 100) { "Max steps must exceed the [100, maxSteps) tail bucket lower bound" }

  val recordWords = MEM_BASE + MEM_WORDS + 1
  require(file.length() % (recordWords * Int.SIZE_BYTES).toLong() == 0L) {
    "File does not contain whole VM records: bytes=${file.length()}, recordWords=$recordWords"
  }

  val availableVms = (file.length() / (recordWords * Int.SIZE_BYTES).toLong()).toInt()
  val vmCount = min(requestedVmCount ?: availableVms, availableVms)
  require(vmCount > 0) { "VM count must be positive" }

  println("file=${file.path}")
  println("run: vmCount=$vmCount chunkVms=$chunkVms workers=$workers quantum=$quantum maxSteps=$maxSteps")

  val stats = HaltingStats(maxSteps)
  val bridge = metalBridge()
  val timerNs = measureNanoTime {
    DataInputStream(BufferedInputStream(FileInputStream(file), 1 shl 20)).use { input ->
      var processed = 0
      while (processed < vmCount) {
        val n = min(chunkVms, vmCount - processed)
        val vmWords = IntArray(n * VM_STRIDE)
        readVmChunk(input, vmWords, n, recordWords, processed)

        val kernelNs = measureNanoTime {
          val ok = bridge.metalRunHypervisor(
            RASP_MSL,
            "run_hypervisor",
            n,
            min(workers, n),
            MEM_WORDS,
            OUTPUT_WORDS,
            quantum,
            maxSteps,
            vmWords
          )
          require(ok != 0) { "metalRunHypervisor failed" }
        }

        stats.add(vmWords, n)
        processed += n
        println(
          "chunk processed=$processed/$vmCount kernel=${formatSeconds(kernelNs)}s " +
            "halted=${stats.naturalHalts} tail100=${stats.tail100ToTimeout} timeout=${stats.timeoutInfinity}"
        )
      }
    }
  }

  stats.writeOutputs(File("build/haltingDistribution"), vmCount)
  println("elapsed=${formatSeconds(timerNs)}s rate=${formatRate(vmCount, timerNs)} vm/s")
  stats.print(vmCount)
}

private class HaltingStats(private val maxSteps: Int) {
  val stepCounts = LongArray(100)
  var tail100ToTimeout = 0L
  var timeoutInfinity = 0L
  var nonHalted = 0L
  var naturalHalts = 0L

  fun add(vmWords: IntArray, vmCount: Int) {
    var vm = 0
    while (vm < vmCount) {
      val base = vm * VM_STRIDE
      val halted = vmWords[base + HALTED] != 0
      val steps = vmWords[base + STEPS]

      when {
        halted && steps < 100 -> {
          naturalHalts++
          stepCounts[steps]++
        }

        halted && steps < maxSteps -> {
          naturalHalts++
          tail100ToTimeout++
        }

        steps >= maxSteps -> timeoutInfinity++
        else -> nonHalted++
      }
      vm++
    }
  }

  fun print(vmCount: Int) {
    println()
    println("halting histogram for brk grammar:")
    println("  total=$vmCount")
    println("  naturalHalts=$naturalHalts")
    println("  tau_h in [100,$maxSteps): $tail100ToTimeout")
    println("  $maxSteps <= tau_h ~= infinity: $timeoutInfinity")
    if (nonHalted > 0) println("  nonHaltedBeforeTimeoutMark=$nonHalted")
    println("  pgf small buckets:")
    println(pgfSmallCoordinates())
    println("  pgf tail [100,$maxSteps):")
    println("    (100,$tail100ToTimeout)")
    println("  pgf timeout:")
    println("    (101,$timeoutInfinity)")
  }

  fun writeOutputs(directory: File, vmCount: Int) {
    directory.mkdirs()
    directory.resolve("halting_distribution_brk.tsv").printWriter().use { out ->
      out.println("bucket\tcount")
      var step = 1
      while (step < 100) {
        out.println("$step\t${stepCounts[step]}")
        step++
      }
      out.println("[100,$maxSteps)\t$tail100ToTimeout")
      out.println("[$maxSteps,infinity]\t$timeoutInfinity")
      out.println("naturalHalts\t$naturalHalts")
      out.println("total\t$vmCount")
    }

    directory.resolve("halting_distribution_brk.pgf").writeText(
      buildString {
        appendLine("% brk grammar, total=$vmCount, tau_max=$maxSteps")
        appendLine("% small buckets: tau_h = 1..99")
        appendLine("\\addplot+[fill=lightgray] coordinates {")
        appendLine("  ${pgfSmallCoordinates()}")
        appendLine("};")
        appendLine("\\addplot+[fill=oiOrange, draw=oiOrange, bar width=0.7pt] coordinates { (100,$tail100ToTimeout) };")
        appendLine("\\addplot+[fill=oiVermilion, draw=oiVermilion, bar width=0.7pt] coordinates { (101,$timeoutInfinity) };")
      }
    )
  }

  private fun pgfSmallCoordinates(): String =
    (1 until 100).joinToString(" ") { "($it,${stepCounts[it]})" }
}

private fun readVmChunk(input: DataInputStream, vmWords: IntArray, vmCount: Int, recordWords: Int, offset: Int) {
  var vm = 0
  while (vm < vmCount) {
    val base = vm * VM_STRIDE
    var i = 0
    while (i < VM_STRIDE) {
      vmWords[base + i] = input.readInt()
      i++
    }

    val terminator = input.readInt()
    require(terminator == Int.MAX_VALUE) {
      "VM ${offset + vm}: expected terminator ${Int.MAX_VALUE}, got $terminator"
    }
    require(i + 1 == recordWords) { "Unexpected record layout: read ${i + 1} words, expected $recordWords" }
    vm++
  }
}

private fun stringSetting(propertyName: String, envName: String, default: String): String =
  System.getProperty(propertyName) ?: System.getenv(envName) ?: default

private fun intSetting(propertyName: String, envName: String, default: Int): Int =
  optionalIntSetting(propertyName, envName) ?: default

private fun optionalIntSetting(propertyName: String, envName: String): Int? =
  System.getProperty(propertyName)?.toIntOrNull() ?: System.getenv(envName)?.toIntOrNull()

private fun formatSeconds(ns: Long): String = "%.3f".format(Locale.US, ns / 1e9)

private fun formatRate(vmCount: Int, ns: Long): String {
  val secs = ns / 1e9
  return "%.1f".format(Locale.US, if (secs > 0.0) vmCount / secs else Double.POSITIVE_INFINITY)
}
