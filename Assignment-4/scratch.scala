import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.util.Random

object AGMCommunityDetection {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("BigCLAM Community Detection").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val nodes = "/Users/arnavsingh/Downloads/git_web_ml/musae_git_target.csv"
    val numCommunities = 6 
    val nodesRDD: RDD[(VertexId, Array[Double])] = spark.read
      .option("header", "true")
      .csv(nodes)
      .rdd
      .map(row => (row.getString(0).toLong, Array.fill(numCommunities)(Random.nextDouble())))

    val edges = "/Users/arnavsingh/Downloads/git_web_ml/musae_git_edges.csv"
    val edgesRDD: RDD[Edge[Int]] = spark.read
      .option("header", "true")
      .csv(edges)
      .rdd
      .flatMap(row => Seq(
        Edge(row.getString(0).toLong, row.getString(1).toLong, 1),
        Edge(row.getString(1).toLong, row.getString(0).toLong, 1)
      ))

    val graph = Graph(nodesRDD, edgesRDD)
    println(s"Number of Vertices: ${graph.vertices.count()}\nNumber of Edges: ${graph.edges.count()} edges.")

    val maxIterations = 20  
    val learningRate = 0.01

    var memberships: VertexRDD[Array[Double]] = graph.vertices

    for (iter <- 1 to maxIterations) {
      val incomingMessages = graph.aggregateMessages[Array[Double]](
        triplet => {
          triplet.sendToSrc(triplet.dstAttr)
          triplet.sendToDst(triplet.srcAttr)
        },
        (a, b) => a.zip(b).map { case (x, y) => x + y } 
      )

      memberships = memberships.innerJoin(incomingMessages) { (_, current, incoming) =>
        current.zip(incoming).map { case (c, i) =>
          c + learningRate * (i - c) 
        }
      }
    }

    val communities = memberships.mapValues(scores =>
      scores.zipWithIndex.maxBy(_._1)._2 
    )

    println("Detected Communities:")
    communities.collect().foreach(println)

    val modularity = computeModularity(graph, communities)
    println(s"Modularity: $modularity")
  }

  def computeModularity(graph: Graph[Array[Double], Int], communities: VertexRDD[Int]): Double = {
    val m = graph.edges.count()

    val degreesMap = graph.degrees.mapValues(_.toDouble).collectAsMap()
    val broadcastDegrees = graph.vertices.sparkContext.broadcast(degreesMap)

    val communitiesMap = communities.collectAsMap()
    val broadcastCommunities = graph.vertices.sparkContext.broadcast(communitiesMap)

    val modularity = graph.triplets.map { triplet =>
      val srcCommunity = broadcastCommunities.value.getOrElse(triplet.srcId, -1)
      val dstCommunity = broadcastCommunities.value.getOrElse(triplet.dstId, -1)

      val Aij = 1.0 
      val expected = broadcastDegrees.value.getOrElse(triplet.srcId, 0.0) *
        broadcastDegrees.value.getOrElse(triplet.dstId, 0.0) / (2.0 * m)

      if (srcCommunity == dstCommunity) Aij - expected else 0.0
    }.sum() / (2.0 * m)

    broadcastDegrees.unpersist()
    broadcastCommunities.unpersist()
    modularity
  }


}

