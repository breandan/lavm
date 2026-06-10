package org.example

import ai.hypergraph.kaliningraph.automata.completeWithSparseGRE
import ai.hypergraph.kaliningraph.parsing.tmLst
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import kotlin.streams.asStream

fun main() {
  val sampledProgramWords = IntArray(TOTAL_PROGRAMS * MEM_WORDS)
  val sources = mutableListOf<String>()
  val cfg = loopyHeapless

  completeWithSparseGRE(List(120) { "_" }, cfg)!!
    .sampleStrWithoutReplacement(cfg.tmLst).asStream().parallel()
    .map { it to it.compileToRASPBytecode() }
    .filter { it.second.size < 127 }
    .limit(TOTAL_PROGRAMS.toLong())
    .toList()
    .forEachIndexed { vm, (source, bytecode) ->
      sources += source
      packProgramIntoBuffer(bytecode, sampledProgramWords, vm)
    }

  writeVmInitialStates(sampledProgramWords)
}

fun writeVmInitialStates(programWords: IntArray, vmCount: Int = TOTAL_PROGRAMS, file: File = DEFAULT_VM_INITIAL_STATES_FILE) {
  file.parentFile?.mkdirs()
  println("Writing to ${file.absolutePath}")
  DataOutputStream(BufferedOutputStream(FileOutputStream(file))).use { out ->
    var vm = 0
    while (vm < vmCount) {
      var i = 0
      while (i < MEM_BASE) { out.writeInt(0); i++ }

      val src = vm * MEM_WORDS
      i = 0
      while (i < MEM_WORDS) { out.writeInt(programWords[src + i]); i++ }

      out.writeInt(Int.MAX_VALUE)
      vm++
    }
  }
  println("Wrote ${file.length()} bytes to ${file.absolutePath}")
}
