import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.min

// The new algorithms are kind of sketches, just to get the idea and see the benchmarks.
// This is not supposed to go directly to production
object NDArrayDot {

    /*
    This approach adapts the algorithm that works for matrices, that are stored by whole rows, not by rows.
    However, it goes with directly indexing the source arrays, which requires a lot of arithmetic operations,
    so this implementation is actually slow. More than that, it is totally unreadable.
     */
    suspend fun new(a: FloatNDArray, b: FloatNDArray, c: MutableFloatNDArray) {
        val m = a.shape[0]
        val t = b.shape[0]
        val n = b.shape[1]

        val PAGE_BYTES = 4 * 1024
        val PAGE_FLOATS = PAGE_BYTES / Float.SIZE_BYTES

        val mts: Int
        val tts: Int
        val nts = PAGE_FLOATS // 1024

        // these numbers are carefully calculated, with the implication of L2 cache being at least 256 KiB (mostly true)
        if (m / t >= 10) {
            mts = 256
            tts = 24
        } else {
            mts = 24
            tts = 30
        }

        val ablocks = a.array.blocks
        val bblocks = b.array.blocks
        val cblocks = c.array.blocks

        val aBlockSize = a.array.blockSize
        val bBlockSize = c.array.blockSize

        val aBlocksInRow = a.shape[1] / aBlockSize
        val bBlocksInRow = b.shape[1] / bBlockSize

        for (it in 0 until m step mts) {
            val ie = min(it, m - mts) + mts
            for (kt in 0 until t step tts) {
                val ke = min(kt, t - tts) + tts

                val kb = kt / aBlockSize
                val kbOff = kt % aBlockSize

                val atOffset = it * aBlocksInRow + kb

                for (jt in 0 until n step nts) {
                    val je = min(jt, n - nts) + nts

                    val jb = jt / bBlockSize
                    val jbOff = jt % bBlockSize

                    var crOffset = it * bBlocksInRow + jb
                    var arOffset = atOffset

                    val btOffset = kt * bBlocksInRow + jb

                    for (i in it until ie) {
                        var k = kt
                        var kk = kbOff

                        var aOffset = arOffset
                        var brOffset = btOffset

                        while (k < ke) {
                            val ab = ablocks[aOffset]

                            while (kk < aBlockSize && k < ke) {
                                val aik = ab[kk]

                                var j = jt
                                var jj = jbOff

                                var bOffset = brOffset
                                var cOffset = crOffset

                                while (j < je) {
                                    val bb = bblocks[bOffset]
                                    val cb = cblocks[cOffset]

                                    while (jj < bBlockSize && j < je) {
                                        cb[jj] += aik * bb[jj]
                                        jj++; j++
                                    }
                                    jj = 0; bOffset++; cOffset++
                                }
                                kk++; k++; brOffset += bBlocksInRow
                            }
                            kk = 0; aOffset++
                        }

                        arOffset += aBlocksInRow; crOffset += bBlocksInRow
                    } // end for i
                }
            }
        }
    }

    /*
    This function allocates new matrices, that are stored by whole rows, then it applies the best algorithm for it.
    However, when either of matrices is large horizontally, allocation of large arrays consumes quite a lot of time.
     */
    suspend fun copy(a: FloatNDArray, b: FloatNDArray, c_: MutableFloatNDArray) {
        val a = a.toFloatArray()
        val b = b.toFloatArray()
        val c = Array(a.size) { FloatArray(b[0].size) }

        val m = a.size
        val t = b.size
        val n = b[0].size

        val PAGE_BYTES = 4 * 1024
        val PAGE_FLOATS = PAGE_BYTES / Float.SIZE_BYTES

        val mts: Int
        val tts: Int
        val nts = PAGE_FLOATS // 1024

        // these numbers are carefully calculated, with the implication of L2 cache being at least 256 KiB (mostly true)
        if (m / t >= 10) {
            mts = 256
            tts = 24
        } else {
            mts = 24
            tts = 30
        }

        for (it in 0 until m step mts) {
            val ie = min(it, m - mts) + mts
            for (kt in 0 until t step tts) {
                val ke = min(kt, t - tts) + tts
                for (jt in 0 until n step nts) {
                    val je = min(jt, n - nts) + nts
                    for (i in it until ie) {
                        val ci = c[i]
                        val ai = a[i]
                        for (k in kt until ke) {
                            val bk = b[k]
                            val aik = ai[k]
                            for (j in jt until je) {
                                ci[j] += aik * bk[j]
                            }
                        }
                    }
                }
            }
        }

        val cbs = c_.array.blockSize
        val cbnr = c_.array.blocksNum / m
        for (i in 0 until m) {
            val ci = c[i]
            val offset = i * cbnr
            for (jb in 0 until cbnr) {
                val destination = c_.array.blocks[offset + jb]
                val startIndex = jb * cbs
                ci.copyInto(destination, 0, startIndex, startIndex + destination.size)
            }
        }
    }

    /*
    This function allocates new matrices, that are stored by blocks too, but the size of the block is calculated
    for cache-friendliness of this specific algorithm. This approach is mostly best, as it is triple as fast as the
    [old] function on 4096x4096x4096. However, in some cases the old approach works good enough, so that this function
    sometimes fails to overcome the allocation&copying overhead. See comments in benchmarks.
     */
    suspend fun resize(a_: FloatNDArray, b_: FloatNDArray, c_: MutableFloatNDArray) {
        val m = a_.shape[0]
        val t = b_.shape[0]
        val n = b_.shape[1]

        val PAGE_BYTES = 4 * 1024
        val PAGE_FLOATS = PAGE_BYTES / Float.SIZE_BYTES

        val mts: Int
        val tts: Int
        val nts = PAGE_FLOATS // 1024

        // these numbers are carefully calculated, with the implication of L2 cache being at least 256 KiB (mostly true)
        if (m / t >= 10) {
            mts = 256
            tts = 24
        } else {
            mts = 24
            tts = 30
        }

        val (a, aBlocksInRow) = if (a_.array.blockSize > tts) {
            emptyBlocks(a_.shape, blockSize = tts)
        } else {
            a_.array.blocks to (a_.shape[1] / a_.array.blockSize)
        }
        val aBlockSize = a[0].size

        val (b, bBlocksInRow) = if (b_.array.blockSize > nts) {
            emptyBlocks(b_.shape, blockSize = nts)
        } else {
            b_.array.blocks to (b_.shape[1] / b_.array.blockSize)
        }
        val bBlockSize = b[0].size

        val c = if (c_.array.blockSize > nts) {
            emptyBlocks(c_.shape, blockSize = nts).first
        } else {
            c_.array.blocks
        }


//        val (a, aBlocksInRow) = emptyBlocks(a_.shape, blockSize = tts)
//        val (b, bBlocksInRow) = emptyBlocks(b_.shape, blockSize = nts)
//        val (c, _) = emptyBlocks(c_.shape, blockSize = nts)

        copyBlocks(a_.array.blocks, a)
        copyBlocks(b_.array.blocks, b)

        for (it in 0 until m step mts) {
            val ie = min(it, m - mts) + mts
            for (kt in 0 until aBlocksInRow) {
                for (jt in 0 until bBlocksInRow) {
                    for (i in it until ie) {
                        val ci = c[i * bBlocksInRow + jt]
                        val ai = a[i * aBlocksInRow + kt]
                        for (k in ai.indices) {
                            val bk = b[(kt * aBlockSize + k) * bBlocksInRow + jt]
                            val aik = ai[k]
                            for (j in ci.indices) {
                                ci[j] += aik * bk[j]
                            }
                        }
                    }
                }
            }
        }

        copyBlocks(c, c_.array.blocks)
    }


    suspend fun old(a: FloatNDArray, b: FloatNDArray, c: MutableFloatNDArray) {
//        a.dot(b, c, EmptyCoroutineContext)

        require(a.shape.size in 1..2 && b.shape.size in 1..2)
        val actualThis = (if (a.shape.size == 1) a.reshape(intArrayOf(1, a.shape[0])) else a) as FloatNDArray
        val actualOther = (if (b.shape.size == 1) b.reshape(intArrayOf(1, b.shape[0])) else b) as FloatNDArray

        require(actualThis.shape[1] == actualOther.shape[0])

        val n = actualThis.shape[0]
//        val t = actualThis.shape[1]

        val lBlockSize = actualThis.array.blockSize
        val rdBlockSize = c.array.blockSize

        val lBlocksInRow = a.shape[1] / lBlockSize
        val rdBlocksInRow = b.shape[1] / rdBlockSize

        for (rdCol in 0 until rdBlocksInRow) {
            for (i in 0 until n) {
                /*
                i * rdBlockInRow equals taking i-th line in destination matrix
                rdCol is number of current block in row
                 */
                val destBlock = c.array.blocks[i * rdBlocksInRow + rdCol]
                //i * lBlocksInRow equals taking i-th line in left matrix
                val leftBlockOffset = i * lBlocksInRow
                // iterating over blocks in i-th line in left matrix
                for (lCol in 0 until lBlocksInRow) {
                    val leftBlock = actualThis.array.blocks[leftBlockOffset + lCol]
                    val rightBlockOffset = lCol * lBlockSize

                    // iterating in left block
                    for (k in 0 until lBlockSize) {
                        val temp = leftBlock[k]
                        /*
                         * lCol * lBlockSize + k is linear index in row in left matrix
                         * number temp staying at [i, lCol * lBlockSize + k] in left matrix,
                         * therefore, we should take (lCol * lBlockSize + k) row in right matrix
                         * (lCol * lBlockSize) moved in rightBlockOffset due to performance purposes
                         */
                        val rightBlock = actualOther.array.blocks[(rightBlockOffset + k) * rdBlocksInRow + rdCol]

                        for (j in 0 until rdBlockSize) {
                            destBlock[j] = (destBlock[j] + temp * rightBlock[j]).toFloat()
                        }
                    }
                }
            }
        }
    }

    suspend fun resize_parallel(a_: FloatNDArray, b_: FloatNDArray, c_: MutableFloatNDArray) {
        val m = a_.shape[0]
        val t = b_.shape[0]
        val n = b_.shape[1]

        val PAGE_BYTES = 4 * 1024
        val PAGE_FLOATS = PAGE_BYTES / Float.SIZE_BYTES

        val mts: Int
        val tts: Int
        val nts = PAGE_FLOATS // 1024

        // these numbers are carefully calculated, with the implication of L2 cache being at least 256 KiB (mostly true)
        if (m / t >= 10) {
            mts = 256
            tts = 24
        } else {
            mts = 24
            tts = 30
        }

        val (a, aBlocksInRow) = if (a_.array.blockSize > tts) {
            emptyBlocks(a_.shape, blockSize = tts)
        } else {
            a_.array.blocks to (a_.shape[1] / a_.array.blockSize)
        }
        val aBlockSize = a[0].size

        val (b, bBlocksInRow) = if (b_.array.blockSize > nts) {
            emptyBlocks(b_.shape, blockSize = nts)
        } else {
            b_.array.blocks to (b_.shape[1] / b_.array.blockSize)
        }
        val bBlockSize = b[0].size

        val c = if (c_.array.blockSize > nts) {
            emptyBlocks(c_.shape, blockSize = nts).first
        } else {
            c_.array.blocks
        }

        copyBlocks(a_.array.blocks.asSequence().chunked(a_.shape[1] / a_.array.blockSize) { it.toTypedArray() }, a.asSequence().chunked(aBlocksInRow) { it.toTypedArray() })
        copyBlocks(b_.array.blocks.asSequence().chunked(b_.shape[1] / b_.array.blockSize) { it.toTypedArray() }, b.asSequence().chunked(bBlocksInRow) { it.toTypedArray() })

        coroutineScope {
            for (it in 0 until m step mts) {
                val ie = min(it, m - mts) + mts
                for (jt in 0 until bBlocksInRow) launch {
                    for (kt in 0 until aBlocksInRow) {
                        for (i in it until ie) {
                            val ci = c[i * bBlocksInRow + jt]
                            val ai = a[i * aBlocksInRow + kt]
                            for (k in ai.indices) {
                                val bk = b[(kt * aBlockSize + k) * bBlocksInRow + jt]
                                val aik = ai[k]
                                for (j in ci.indices) {
                                    ci[j] += aik * bk[j]
                                }
                            }
                        }
                    }
                }
            }
        }

        copyBlocks(c.asSequence().chunked(bBlocksInRow) { it.toTypedArray() }, c_.array.blocks.asSequence().chunked(a_.shape[1] / a_.array.blockSize) { it.toTypedArray() })
    }

    suspend fun resize_parallel_lesslaunches(a_: FloatNDArray, b_: FloatNDArray, c_: MutableFloatNDArray) {
        val m = a_.shape[0]
        val t = b_.shape[0]
        val n = b_.shape[1]

        val PAGE_BYTES = 4 * 1024
        val PAGE_FLOATS = PAGE_BYTES / Float.SIZE_BYTES

        val mts: Int
        val tts: Int
        val nts = PAGE_FLOATS // 1024

        // these numbers are carefully calculated, with the implication of L2 cache being at least 256 KiB (mostly true)
        if (m / t >= 10) {
            mts = 256
            tts = 24
        } else {
            mts = 24
            tts = 30
        }

        val (a, aBlocksInRow) = if (a_.array.blockSize > tts) {
            emptyBlocks(a_.shape, blockSize = tts)
        } else {
            a_.array.blocks to (a_.shape[1] / a_.array.blockSize)
        }
        val aBlockSize = a[0].size

        val (b, bBlocksInRow) = if (b_.array.blockSize > nts) {
            emptyBlocks(b_.shape, blockSize = nts)
        } else {
            b_.array.blocks to (b_.shape[1] / b_.array.blockSize)
        }
        val bBlockSize = b[0].size

        val c = if (c_.array.blockSize > nts) {
            emptyBlocks(c_.shape, blockSize = nts).first
        } else {
            c_.array.blocks
        }


//        val (a, aBlocksInRow) = emptyBlocks(a_.shape, blockSize = tts)
//        val (b, bBlocksInRow) = emptyBlocks(b_.shape, blockSize = nts)
//        val (c, _) = emptyBlocks(c_.shape, blockSize = nts)

        copyBlocks(a_.array.blocks.asSequence().chunked(a_.shape[1] / a_.array.blockSize) { it.toTypedArray() }, a.asSequence().chunked(aBlocksInRow) { it.toTypedArray() })
        copyBlocks(b_.array.blocks.asSequence().chunked(b_.shape[1] / b_.array.blockSize) { it.toTypedArray() }, b.asSequence().chunked(bBlocksInRow) { it.toTypedArray() })

        val cores = Runtime.getRuntime().availableProcessors()
        val mTiles = (m + mts - 1) / mts
        val nTiles = bBlocksInRow
        val mParallelChunks = 1 + cores * mTiles / (mTiles + nTiles)
        val nParallelChunks = (cores + mParallelChunks - 1) / mParallelChunks
        val mTilesPerChunk = (mTiles + mParallelChunks - 1) / mParallelChunks
        val nTilesPerChunk = (nTiles + nParallelChunks - 1) / nParallelChunks
        val mChunkSize = mts * mTilesPerChunk

        coroutineScope {
            for (ic in 0 until m step mChunkSize) {
                for (jc in 0 until nTiles step nTilesPerChunk) {
                    val jte = minOf(jc + nTilesPerChunk, nTiles)

                    launch {
                        for (it in ic until ic + mChunkSize step mts) {
                            val ie = minOf(it, m - mts) + mts
                            for (kt in 0 until aBlocksInRow) {
                                for (jt in jc until jte) {

                                    for (i in it until ie) {
                                        val ci = c[i * bBlocksInRow + jt]
                                        val ai = a[i * aBlocksInRow + kt]
                                        for (k in ai.indices) {
                                            val bk = b[(kt * aBlockSize + k) * bBlocksInRow + jt]
                                            val aik = ai[k]
                                            for (j in ci.indices) {
                                                ci[j] += aik * bk[j]
                                            }
                                        }
                                    }

                                }
                            }
                        }
                    }

                }
            }
        }

        copyBlocks(c.asSequence().chunked(bBlocksInRow) { it.toTypedArray() }, c_.array.blocks.asSequence().chunked(a_.shape[1] / a_.array.blockSize) { it.toTypedArray() })
    }

    suspend fun old_parallel(a: FloatNDArray, b: FloatNDArray, c: MutableFloatNDArray) = coroutineScope {
//        a.dot(b, c, EmptyCoroutineContext)

        require(a.shape.size in 1..2 && b.shape.size in 1..2)
        val actualThis = (if (a.shape.size == 1) a.reshape(intArrayOf(1, a.shape[0])) else a) as FloatNDArray
        val actualOther = (if (b.shape.size == 1) b.reshape(intArrayOf(1, b.shape[0])) else b) as FloatNDArray

        require(actualThis.shape[1] == actualOther.shape[0])

        val n = actualThis.shape[0]
//        val t = actualThis.shape[1]

        val lBlockSize = actualThis.array.blockSize
        val rdBlockSize = c.array.blockSize

        val lBlocksInRow = a.shape[1] / lBlockSize
        val rdBlocksInRow = b.shape[1] / rdBlockSize

        for (rdCol in 0 until rdBlocksInRow) {
            launch {
                for (i in 0 until n) {
                    /*
                    i * rdBlockInRow equals taking i-th line in destination matrix
                    rdCol is number of current block in row
                     */
                    val destBlock = c.array.blocks[i * rdBlocksInRow + rdCol]
                    //i * lBlocksInRow equals taking i-th line in left matrix
                    val leftBlockOffset = i * lBlocksInRow
                    // iterating over blocks in i-th line in left matrix
                    for (lCol in 0 until lBlocksInRow) {
                        val leftBlock = actualThis.array.blocks[leftBlockOffset + lCol]
                        val rightBlockOffset = lCol * lBlockSize

                        // iterating in left block
                        for (k in 0 until lBlockSize) {
                            val temp = leftBlock[k]
                            /*
                             * lCol * lBlockSize + k is linear index in row in left matrix
                             * number temp staying at [i, lCol * lBlockSize + k] in left matrix,
                             * therefore, we should take (lCol * lBlockSize + k) row in right matrix
                             * (lCol * lBlockSize) moved in rightBlockOffset due to performance purposes
                             */
                            val rightBlock = actualOther.array.blocks[(rightBlockOffset + k) * rdBlocksInRow + rdCol]

                            for (j in 0 until rdBlockSize) {
                                destBlock[j] = (destBlock[j] + temp * rightBlock[j]).toFloat()
                            }
                        }
                    }
                }
            }
        }
    }

    suspend fun cupertankParallel_iterator(left: FloatNDArray, right: FloatNDArray, dest: MutableFloatNDArray) {
        val n = left.shape[0]
        val k = left.shape[1]
        val m = right.shape[1]

        val lBlockSize = left.array.blockSize
        val rdBlockSize = right.array.blockSize

        val lBlocksInRow = left.shape[1] / lBlockSize
        val rdBlocksInRow = right.shape[1] / rdBlockSize

        val threads = Runtime.getRuntime().availableProcessors()
        val nStep = if (n < threads) 1 else n / threads

        coroutineScope {
            val leftBlocks = left.array.blocks
            val rightBlocks = right.array.blocks
            val destBlocks = dest.array.blocks

            for (nStart in 0 until n step nStep) {
                launch {
                    for (i in nStart until min(nStart + nStep, n)) {
                        val leftBlockOffset = i * lBlocksInRow
                        val destBlockOffset = i * rdBlocksInRow
                        val rightBlocksIter = rightBlocks.iterator()

                        for (lCol in 0 until lBlocksInRow) {
                            val leftBlock = leftBlocks[leftBlockOffset + lCol]

                            for (k in 0 until lBlockSize) {
                                val temp = leftBlock[k]

                                for (rdCol in 0 until rdBlocksInRow) {
                                    val destBlock = destBlocks[destBlockOffset + rdCol]
                                    val rightBlock = rightBlocksIter.next()

                                    for (j in destBlock.indices) {
                                        destBlock[j] += temp * rightBlock[j]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}