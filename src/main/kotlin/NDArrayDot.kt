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

    suspend fun some_shit_2(a: FloatNDArray, b: FloatNDArray, c: MutableFloatNDArray) {
        //        a.dot(b, c, EmptyCoroutineContext)

        require(a.shape.size in 1..2 && b.shape.size in 1..2)
        val actualThis = (if (a.shape.size == 1) a.reshape(intArrayOf(1, a.shape[0])) else a) as FloatNDArray
        val actualOther = (if (b.shape.size == 1) b.reshape(intArrayOf(1, b.shape[0])) else b) as FloatNDArray

        require(actualThis.shape[1] == actualOther.shape[0])

        val n = actualThis.shape[0]
        val t = actualThis.shape[1]

        val lBlockSize = actualThis.array.blockSize
        val rdBlockSize = c.array.blockSize

        val lBlocksInRow = a.shape[1] / lBlockSize
        val rdBlocksInRow = b.shape[1] / rdBlockSize

        val nStep = 2
        val tTileSize = minOf(lBlockSize, ((65024 - nStep * rdBlockSize) / (nStep + rdBlockSize)).takeIf { it > 0 } ?: t)

        coroutineScope {
            for (iStart in 0 until n step nStep) {
                val iEnd = minOf(iStart + nStep, n)
                for (rdCol in 0 until rdBlocksInRow) {
                    launch {
                        // iterating over blocks in i-th line in left matrix
                        for (lCol in 0 until lBlocksInRow) {
                            for (tStart in 0 until lBlockSize step tTileSize) {
                                val tEnd = minOf(tStart + tTileSize, lBlockSize)
                                /*
                                        i * rdBlockInRow equals taking i-th line in destination matrix
                                        rdCol is number of current block in row
                                         */
                                for (i in iStart until iEnd) {
                                    val destBlock = c.array.blocks[i * rdBlocksInRow + rdCol]
                                    //i * lBlocksInRow equals taking i-th line in left matrix
                                    val leftBlock = actualThis.array.blocks[i * lBlocksInRow + lCol]
                                    val rightBlockOffset = lCol * lBlockSize

                                    // iterating in left block
                                    for (k in tStart until tEnd) {
                                        val temp = leftBlock[k]
                                        /*
                                                         * lCol * lBlockSize + k is linear index in row in left matrix
                                                         * number temp staying at [i, lCol * lBlockSize + k] in left matrix,
                                                         * therefore, we should take (lCol * lBlockSize + k) row in right matrix
                                                         * (lCol * lBlockSize) moved in rightBlockOffset due to performance purposes
                                                         */
                                        val rightBlock = actualOther.array.blocks[(rightBlockOffset + k) * rdBlocksInRow + rdCol]

                                        for (j in 0 until rdBlockSize) {
                                            destBlock[j] += destBlock[j] + temp * rightBlock[j]
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

    suspend fun some_shit_4(a: FloatNDArray, b: FloatNDArray, c: MutableFloatNDArray) {
        //        a.dot(b, c, EmptyCoroutineContext)

        require(a.shape.size in 1..2 && b.shape.size in 1..2)
        val actualThis = (if (a.shape.size == 1) a.reshape(intArrayOf(1, a.shape[0])) else a) as FloatNDArray
        val actualOther = (if (b.shape.size == 1) b.reshape(intArrayOf(1, b.shape[0])) else b) as FloatNDArray

        require(actualThis.shape[1] == actualOther.shape[0])

        val n = actualThis.shape[0]
        val t = actualThis.shape[1]

        val lBlockSize = actualThis.array.blockSize
        val rdBlockSize = c.array.blockSize

        val lBlocksInRow = a.shape[1] / lBlockSize
        val rdBlocksInRow = b.shape[1] / rdBlockSize

        val nStep = 4
        val tTileSize = minOf(lBlockSize, ((65024 * 2 / 3 - nStep * rdBlockSize) / (nStep + rdBlockSize) - 64 / Float.SIZE_BYTES).takeIf { it > 0 } ?: t)
        // ((512 * 112) + 112 + 512) * 4

        coroutineScope {
            for (iStart in 0 until n step nStep) {
                val iEnd = minOf(iStart + nStep, n)
                for (rdCol in 0 until rdBlocksInRow) {
                    launch {
                        // iterating over blocks in i-th line in left matrix
                        for (lCol in 0 until lBlocksInRow) {
                            for (tStart in 0 until lBlockSize step tTileSize) {
                                val tEnd = minOf(tStart + tTileSize, lBlockSize)
                                /*
                                        i * rdBlockInRow equals taking i-th line in destination matrix
                                        rdCol is number of current block in row
                                         */
                                for (i in iStart until iEnd) {
                                    //i * lBlocksInRow equals taking i-th line in left matrix
                                    val leftBlock = actualThis.array.blocks[i * lBlocksInRow + lCol]
                                    val destBlock = c.array.blocks[i * rdBlocksInRow + rdCol]
                                    val rightBlockOffset = lCol * lBlockSize

                                    // iterating in left block
                                    for (k in tStart until tEnd) {
                                        val rightBlock = actualOther.array.blocks[(rightBlockOffset + k) * rdBlocksInRow + rdCol]
                                        val temp = leftBlock[k]
                                        /*
                                                         * lCol * lBlockSize + k is linear index in row in left matrix
                                                         * number temp staying at [i, lCol * lBlockSize + k] in left matrix,
                                                         * therefore, we should take (lCol * lBlockSize + k) row in right matrix
                                                         * (lCol * lBlockSize) moved in rightBlockOffset due to performance purposes
                                                         */

                                        for (j in 0 until rdBlockSize) {
                                            destBlock[j] += destBlock[j] + temp * rightBlock[j]
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

    suspend fun some_shit_8(a: FloatNDArray, b: FloatNDArray, c: MutableFloatNDArray) {
        //        a.dot(b, c, EmptyCoroutineContext)

        require(a.shape.size in 1..2 && b.shape.size in 1..2)
        val actualThis = (if (a.shape.size == 1) a.reshape(intArrayOf(1, a.shape[0])) else a) as FloatNDArray
        val actualOther = (if (b.shape.size == 1) b.reshape(intArrayOf(1, b.shape[0])) else b) as FloatNDArray

        require(actualThis.shape[1] == actualOther.shape[0])

        val n = actualThis.shape[0] // 24
        val t = actualThis.shape[1] // 256

        val lBlockSize = actualThis.array.blockSize // 256
        val rdBlockSize = c.array.blockSize // 512

        val lBlocksInRow = a.shape[1] / lBlockSize // 1
        val rdBlocksInRow = b.shape[1] / rdBlockSize // 2

        val nStep = 8
        val tTileSize = minOf(lBlockSize, ((65024 - nStep * rdBlockSize) / (nStep + rdBlockSize) - 64 / Float.SIZE_BYTES).takeIf { it > 0 } ?: t)
        // 117



        coroutineScope {
            for (iStart in 0 until n step nStep) {
                val iEnd = minOf(iStart + nStep, n)
                for (rdCol in 0 until rdBlocksInRow) {
                    launch {
                        // iterating over blocks in i-th line in left matrix
                        for (lCol in 0 until lBlocksInRow) {
                            for (tStart in 0 until lBlockSize step tTileSize) {
                                val tEnd = minOf(tStart + tTileSize, lBlockSize)
                                /*
                                        i * rdBlockInRow equals taking i-th line in destination matrix
                                        rdCol is number of current block in row
                                         */
                                for (i in iStart until iEnd) {
                                    val destBlock = c.array.blocks[i * rdBlocksInRow + rdCol]
                                    //i * lBlocksInRow equals taking i-th line in left matrix
                                    val leftBlock = actualThis.array.blocks[i * lBlocksInRow + lCol]
                                    val rightBlockOffset = lCol * lBlockSize

                                    // iterating in left block
                                    for (k in tStart until tEnd) {
                                        val temp = leftBlock[k]
                                        /*
                                                         * lCol * lBlockSize + k is linear index in row in left matrix
                                                         * number temp staying at [i, lCol * lBlockSize + k] in left matrix,
                                                         * therefore, we should take (lCol * lBlockSize + k) row in right matrix
                                                         * (lCol * lBlockSize) moved in rightBlockOffset due to performance purposes
                                                         */
                                        val rightBlock = actualOther.array.blocks[(rightBlockOffset + k) * rdBlocksInRow + rdCol]

                                        for (j in 0 until rdBlockSize) {
                                            destBlock[j] += destBlock[j] + temp * rightBlock[j]
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

    suspend fun cupertankParallel_tilethalf_tilen(left: FloatNDArray, right: FloatNDArray, dest: MutableFloatNDArray) {
        val n = left.shape[0] // 24
        val k = left.shape[1] // 256
        val m = right.shape[1] // 4838

        val lBlockSize = left.array.blockSize // 256
        val rdBlockSize = right.array.blockSize // 2419

        val lBlocksInRow = left.shape[1] / lBlockSize // 1
        val rdBlocksInRow = right.shape[1] / rdBlockSize // 2

        val threads = Runtime.getRuntime().availableProcessors() // 12
        val nStep = if (n < threads) 1 else n / threads * 2 // 2

        val (mBlocksStep, mStep) = run {
            val PAGE_BYTES = 4 * 1024
            val PAGE_FLOATS = PAGE_BYTES / Float.SIZE_BYTES

            val CACHE_LINE_BYTES = 64
            val CACHE_LINE_FLOATS = CACHE_LINE_BYTES / Float.SIZE_BYTES

            if (rdBlockSize > PAGE_FLOATS) {
                1 to PAGE_FLOATS
            } else {
                PAGE_FLOATS / rdBlockSize to rdBlockSize +
                        if (rdBlockSize % CACHE_LINE_FLOATS != 0)
                            CACHE_LINE_FLOATS - (rdBlockSize % CACHE_LINE_FLOATS)
                        else 0
            }
        } // 1, 1024

        val kStep = run {
            val CACHE_BYTES = 256 * 1024 / 2 - 8 * 1024
            val CACHE_FLOATS = CACHE_BYTES / Float.SIZE_BYTES

            /*
            kStep * mStep * mBlockStep + kStep + kStep * mBlockStep + mStep * mBlockStep + mBlockStep = CACHE_FLOATS
            (CACHE_FLOATS - mStep * mBlockStep - mBlockStep) / (mStep * mBlockStep + mBlockStep + 1)

            (30720 - 1024 * 1 - 1 - 32) / (1024 * 1 + 1 + 1)
             */

            val CACHE_LINE_BYTES = 64
            val CACHE_LINE_FLOATS = CACHE_LINE_BYTES / Float.SIZE_BYTES

//            val idealKStep = (CACHE_FLOATS - nStep * (mStep * mBlocksStep) - nStep * mBlocksStep - nStep) / (nStep + mStep * mBlocksStep + mBlocksStep)
            val idealKStep = (CACHE_FLOATS - mStep * mBlocksStep - mBlocksStep - CACHE_LINE_FLOATS * 2) / (mStep * mBlocksStep + mBlocksStep + 1)


            (idealKStep /*- (idealKStep % CACHE_LINE_FLOATS)*/)
                .takeIf { it in 1 until lBlockSize } ?: lBlockSize
        } // 48
        // 30720
        // 27

        coroutineScope {
            for (nStart in 0 until n step nStep) {
                val nEnd = min(nStart + nStep, n)
                for (rdColStart in 0 until rdBlocksInRow step mBlocksStep) {
                    val rdColEnd = minOf(rdColStart + mBlocksStep, rdBlocksInRow)
                    for (jStart in 0 until rdBlockSize step mStep) {
                        val jEnd = minOf(jStart + mStep, rdBlockSize)
                        launch {
                            for (lCol in 0 until lBlocksInRow) {
                                val rightBlockOffset = lCol * lBlockSize
                                for (kStart in 0 until lBlockSize step kStep) {
                                    val kEnd = minOf(kStart + kStep, lBlockSize)
                                    for (i in nStart until nEnd) {
                                        val leftBlockOffset = i * lBlocksInRow
                                        val destBlockOffset = i * rdBlocksInRow

                                        val leftBlock = left.array.blocks[leftBlockOffset + lCol]

                                        for (k in kStart until kEnd) {
                                            val rightBlockOffsetFull = (rightBlockOffset + k) * rdBlocksInRow
                                            val temp = leftBlock[k]

                                            for (rdCol in rdColStart until rdColEnd) {
                                                val destBlock = dest.array.blocks[destBlockOffset + rdCol]
                                                val rightBlock = right.array.blocks[rightBlockOffsetFull + rdCol]

                                                for (j in jStart until jEnd) {
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
        }
    }

    suspend fun cupertankParallel_shit(left: FloatNDArray, right: FloatNDArray, dest: MutableFloatNDArray) {
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
            for (nStart in 0 until n step nStep) {
                launch {
                    for (i in nStart until min(nStart + nStep, n)) {
                        val leftBlockOffset = i * lBlocksInRow
                        val destBlockOffset = i * rdBlocksInRow
                        val rightBlocksIter = right.array.blocks.iterator()

                        for (lCol in 0 until lBlocksInRow) {
                            val leftBlock = left.array.blocks[leftBlockOffset + lCol]
//                            val rightBlockOffset = lCol * lBlockSize

                            for (k in 0 until lBlockSize) {
//                                val rightBlockOffsetFull = (rightBlockOffset + k) * rdBlocksInRow
                                val temp = leftBlock[k]

                                for (rdCol in 0 until rdBlocksInRow) {
                                    val destBlock = dest.array.blocks[destBlockOffset + rdCol]
//                                    val rightBlock = right.array.blocks[rightBlockOffsetFull + rdCol]
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