plugins {
  kotlin("jvm") version "2.3.10"
  application
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
  mavenLocal()
  mavenCentral()
}

val swiftGenerator by configurations.creating

dependencies {
  implementation("ai.hypergraph:galoisenne:0.2.2")
  implementation("net.java.dev.jna:jna:5.19.0")
  swiftGenerator(kotlin("stdlib"))
  testImplementation(kotlin("test"))
}

kotlin {
  jvmToolchain(21)
}

application {
  mainClass.set("org.example.MainKt")
}

tasks.test {
  useJUnitPlatform()
}

tasks.register<JavaExec>("writeVMs") {
  group = "application"
  description = "Runs WriteVMs.kt"
  classpath = sourceSets["main"].runtimeClasspath
  mainClass.set("org.example.WriteVMsKt")
  dependsOn(tasks.named("classes"))
}

tasks.register<JavaExec>("metalRASP") {
  group = "application"
  description = "Runs the Metal-backed RASP hypervisor benchmark"
  classpath = sourceSets["main"].runtimeClasspath + swiftGenerator
  mainClass.set("org.example.MetalRASPKt")
  dependsOn(tasks.named("classes"))
}

tasks.register<JavaExec>("haltingDistribution") {
  group = "application"
  description = "Runs the brk-grammar VM corpus and prints halting histogram coordinates"
  classpath = sourceSets["main"].runtimeClasspath + swiftGenerator
  mainClass.set("org.example.HaltingDistributionKt")
  systemProperties(System.getProperties().mapKeys { it.key.toString() })
  dependsOn(tasks.named("classes"))
}

tasks.register<JavaExec>("compareJVMvMetal") {
  group = "application"
  description = "Compares JVM parallel-stream RASP execution against Apple Metal"
  classpath = sourceSets["main"].runtimeClasspath + swiftGenerator
  mainClass.set("org.example.CompareJVMvMetalKt")
  dependsOn(tasks.named("classes"))
}
