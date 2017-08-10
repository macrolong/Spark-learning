import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, _}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

import scala.collection.mutable.ArrayBuffer
import scala.util.control._
import String._

import scala.collection.mutable.ArrayBuffer

object PCATest {

    def preprocess(line: String): Vector = {
        var line1: Array[String] = line.split(",").filter((x: String) => x != "null")
        var line2: Array[Double] = new Array[Double](line1.length)
        if (line1.length != 54676)
            line2 = Array(1.0)
        else {
            var count = 0
            val loop = new Breaks
            loop.breakable {
                for (i <- Range(0, 54676)) {
                    try {
                        line2(i) = line1(i).replaceAll("\"", "").toDouble
                    } catch {
                        case ex: java.lang.NumberFormatException => {
                            count = count + 1
                        }
                    }
                    if (count > 1) {
                        line2 = Array(1.0)
                        loop.break
                    }
                }
            }

        }
        Vectors.dense(line2)
    }

    def pearson(dataRDD: RDD[Vector]): CoordinateMatrix = {
        /*
        If we let dataRDD equals the matrix below

                var dataRDD: RDD[Vector] = sc.parallelize(
                    Array(Vectors.dense(1.0, 0.0, 2.0, 0.0),
                          Vectors.dense(2.0, 5.0, 1.0, 3.0),
                          Vectors.dense(4.0 ,5.0 ,6.0 ,1.0))
                    )
        and execute the direction PearsonRDD.toLocalMatrix in scala, it will output:
            scala> PearsonRDD.toLocalMatrix
            res9: org.apache.spark.mllib.linalg.Matrix =
            1.0000000000000002  0.7559289460184544  0.8660254037844388   0.1428571428571428
            0.7559289460184544  1.0                 0.3273268353539886   0.7559289460184545
            0.8660254037844388  0.3273268353539886  1.0                  -0.3711537444790452
            0.1428571428571428  0.7559289460184545  -0.3711537444790452  1.0000000000000002
        which is the same as direction executed in R
                      [,1]      [,2]       [,3]       [,4]
            [1,] 1.0000000 0.7559289  0.8660254  0.1428571
            [2,] 0.7559289 1.0000000  0.3273268  0.7559289
            [3,] 0.8660254 0.3273268  1.0000000 -0.3711537
            [4,] 0.1428571 0.7559289 -0.3711537  1.0000000
        test done!
        */
        val summary: MultivariateStatisticalSummary = Statistics.colStats(dataRDD)
        val mean: Vector = summary.mean
        val variance = summary.variance
        val factor = summary.count - 1.0

        // transform to coordinate Martix in case of OutofMemory
        val coordinateRDD: CoordinateMatrix = new IndexedRowMatrix(dataRDD.zipWithIndex()
          .map((x: (Vector, Long)) => IndexedRow(x._2, x._1))).toBlockMatrix(1024, 1024).toCoordinateMatrix

        // record the coordination of 0 in dataRDD, since method .toCoordinateMatrix will not store 0
        val zeroRDD: CoordinateMatrix = new IndexedRowMatrix(dataRDD
          .map((row: Vector) => Vectors.dense(row.toArray.map(x => if (x == 0) 1.0 else 0))).zipWithIndex()
          .map((x: (Vector, Long)) => IndexedRow(x._2, x._1))).toBlockMatrix(1024, 1024).toCoordinateMatrix

        // padding the negative mean in the position of 0
        val paddingMeanRDD: CoordinateMatrix = new CoordinateMatrix(zeroRDD.entries
          .map((x: MatrixEntry) => MatrixEntry(x.i, x.j, -mean(x.j.toInt))))

        // Centralize the dataRDD
        val CentralizeNotFullRDD: CoordinateMatrix = new CoordinateMatrix(coordinateRDD.entries
          .map((x: MatrixEntry) => MatrixEntry(x.i, x.j, (x.value - mean(x.j.toInt)))))

        val CentralizeRDD: CoordinateMatrix = new CoordinateMatrix(CentralizeNotFullRDD.entries.union(paddingMeanRDD.entries))

        //scale the data RDD
        val NormalizedRDD: CoordinateMatrix = new CoordinateMatrix(CentralizeRDD.entries
          .map((x: MatrixEntry) => MatrixEntry(x.i, x.j, x.value / math.sqrt(variance(x.j.toInt) * factor))))

        // computer the pearson correlation coefficient
        NormalizedRDD.toBlockMatrix.transpose.multiply(NormalizedRDD.toBlockMatrix).toCoordinateMatrix
    }


    def spearman(dataRDD: RDD[Vector]): CoordinateMatrix = {
        /*
        If we let dataRDD equals the matrix below

                var dataRDD: RDD[Vector] = sc.parallelize(
                    Array(Vectors.dense(1.0, 0.0, 2.0, 0.0),
                          Vectors.dense(2.0, 5.0, 1.0, 3.0),
                          Vectors.dense(4.0 ,5.0 ,6.0 ,1.0))
                    )
        and execute the direction PearsonRDD.toLocalMatrix in scala, it will output:
            scala> PearsonRDD.toLocalMatrix
            res31: org.apache.spark.mllib.linalg.Matrix =
            0.9999999999999998  0.8660254037844388  0.4999999999999999   0.4999999999999999
            0.8660254037844388  1.0000000000000002  0.0                  0.8660254037844388
            0.4999999999999999  0.0                 0.9999999999999998   -0.4999999999999999
            0.4999999999999999  0.8660254037844388  -0.4999999999999999  0.9999999999999998

        which is the same as direction executed in R
                      [,1]      [,2]       [,3]       [,4]
            [1,] 1.0000000 0.7559289  0.8660254  0.1428571
            [2,] 0.7559289 1.0000000  0.3273268  0.7559289
            [3,] 0.8660254 0.3273268  1.0000000 -0.3711537
            [4,] 0.1428571 0.7559289 -0.3711537  1.0000000
        test done!
        */

        // ((columnIndex, value), rowUid)
        val colBased = dataRDD.zipWithUniqueId().flatMap{ case (vec, uid) =>
            vec.toArray.view.zipWithIndex.map{ case (v, j) =>
                ((j, v), uid)
            }
        }
        // global sort by (columnIndex, value)
        val sorted = colBased.sortByKey()
        // assign global ranks (using average ranks for tied values)
        val globalRanks = sorted.zipWithIndex().mapPartitions { iter =>
            var preCol = -1
            var preVal = Double.NaN
            var startRank = -1.0
            var cachedUids = ArrayBuffer.empty[Long]
            val flush: () => Iterable[(Long, (Int, Double))] = () => {
                val averageRank = startRank + (cachedUids.size - 1) / 2.0
                val output = cachedUids.map { uid =>
                    (uid, (preCol, averageRank))
                }
                cachedUids.clear()
                output
            }
            iter.flatMap { case (((j, v), uid), rank) =>
                // If we see a new value or cachedUids is too big, we flush ids with their average rank.
                if (j != preCol || v != preVal || cachedUids.size >= 10000000) {
                    val output = flush()
                    preCol = j
                    preVal = v
                    startRank = rank
                    cachedUids += uid
                    output
                } else {
                    cachedUids += uid
                    Iterator.empty
                }
            } ++ flush()
        }
        // Replace values in the input matrix by their ranks compared with values in the same column.
        // Note that shifting all ranks in a column by a constant value doesn't affect result.
        val groupedRanks = globalRanks.groupByKey().map { case (uid, iter) =>
            // sort by column index and then convert values to a vector
            Vectors.dense(iter.toSeq.sortBy(_._1).map(_._2).toArray)
        }
        pearson(groupedRanks)
    }
}

    
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("PCATest")
        val sc = new SparkContext(conf)

        val path = "hdfs:///user/hadoopuser/ExpAltas/Splited/GPL570.Splited00"
        //        var dataRDD: RDD[Vector] = sc.objectFile(path).repartition(args(1).toInt)
        var dataRDD: RDD[Vector] = sc.textFile(path).map(preprocess).filter((row: Vector) => row.size != 1)

        val start = System.nanoTime()

        val PearsonRDD: CoordinateMatrix  = pearson(dataRDD)

        val SpearmanRDD: CoordinateMatrix = spearman(dataRDD)

        //        val rdd_vec: RDD[Vector] = dataRDD.map(row => {
        //          Vectors.dense(row.toArray.zipWithIndex.map(x => {
        //            val a: Double = (x._1 - mean(x._2))/math.sqrt(variance(x._2).toDouble * number)
        //      a
        //          }))
        //        })
        //        val data = new IndexedRowMatrix(rdd_vec.zipWithIndex().map(x=>IndexedRow(x._2, x._1))).toBlockMatrix(1024,1024)
        //    val cov1 = data.transpose.multiply(data).toLocalMatrix

        //compte pearson matrix
        //    val cov1 = data.transpose.multiply(data).toCoordinateMatrix
        

        ////compute pearson matrix
        //    val pearM = new CoordinateMatrix(cov1.entries
        //      .map(x => MatrixEntry(x.i, x.j, if (x.i == x.j) 1.0 else x.value * m /(math.sqrt(v(x.j.toInt)) * math.sqrt(v(x.i.toInt)))))
        //    )
        //println("****vince:result rows:" + pearM.numRows + ", cols:" + pearM.numCols )

        /*
        // validate the result by comparing the data output from Spark 'corr' API on original input
            import breeze.linalg.{DenseMatrix => BDM}
            val matP = BDM.zeros[Double](pearM.numRows.toInt, pearM.numCols.toInt)
            pearM.entries.collect().foreach { case MatrixEntry(i, j, value) =>
              matP(i.toInt, j.toInt) = value
            }
            println(s"****vin:return pearson matrix 1: \n${matP}")

           val correlMatrix: Matrix = Statistics.corr(dataRDD, "pearson")
           println(s"***return spark corr: \n{correlMatrix.toString}")
        */
        println(s"****compute done****")
        val time = (System.nanoTime() - start) / 1e9
        sc.stop()
    }
}
