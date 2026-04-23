package org.example

import ai.hypergraph.kaliningraph.automata.completeWithSparseGRE
import ai.hypergraph.kaliningraph.parsing.tmLst
import ai.hypergraph.kaliningraph.rasp.compileToRASPBytecode
import ai.hypergraph.kaliningraph.repair.loopyHeapless
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