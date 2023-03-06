package com.boris

import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.infra.Blackhole

@Suppress("unused")
@State(Scope.Benchmark)
open class UtilBench {
    @Benchmark
    fun flops_mult_1kk(bh: Blackhole) {
        var i = 0
        var x = 0.99f
        while (i++ < 500_000) {
            x *= x * 1.01f
            Blackhole.consumeCPU(0)
        }
        bh.consume(x)
    }

    @Benchmark
    fun flops_add_1kk(bh: Blackhole) {
        var i = 0
        var x = 0.1f
        while (i++ < 500_000) {
            x += x - 0.099f
            Blackhole.consumeCPU(0)
        }
        bh.consume(x)
    }

    @Benchmark
    fun flops_iter_500k(bh: Blackhole) {
        var i = 0
        while (i++ < 500_000) {
            Blackhole.consumeCPU(0)
        }
        bh.consume(i)
    }
}