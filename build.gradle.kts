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

dependencies {
  implementation("ai.hypergraph:galoisenne:0.2.2")
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