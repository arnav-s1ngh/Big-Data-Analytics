import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.util.Random

object BigCLAMCommunityDetection {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("BigCLAM Community Detection").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    // Load Nodes
    val nodes = "/Users/arnavsingh/Downloads/git_web_ml/musae_git_target.csv"
    val numCommunities = 5 // Number of communities
    val nodesRDD: RDD[(VertexId, Array[Double])] = spark.read
      .option("header", "true")
      .csv(nodes)
      .rdd
      .map(row => (row.getString(0).toLong, Array.fill(numCommunities)(Random.nextDouble())))

    // Load Edges
    val edges = "/Users/arnavsingh/Downloads/git_web_ml/musae_git_edges.csv"
    val edgesRDD: RDD[Edge[Int]] = spark.read
      .option("header", "true")
      .csv(edges)
      .rdd
      .flatMap(row => Seq(
        Edge(row.getString(0).toLong, row.getString(1).toLong, 1),
        Edge(row.getString(1).toLong, row.getString(0).toLong, 1)
      ))

    // Create Graph
    val graph = Graph(nodesRDD, edgesRDD)
    println(s"Number of Vertices: ${graph.vertices.count()}\nNumber of Edges: ${graph.edges.count()} edges.")

    val maxIterations = 20  // Maximum iterations
    val learningRate = 0.01 // Learning rate for optimization

    var memberships: VertexRDD[Array[Double]] = graph.vertices

    // Iteratively update membership scores
    for (iter <- 1 to maxIterations) {
      // Aggregate messages from neighbors
      val incomingMessages = graph.aggregateMessages[Array[Double]](
        triplet => {
          // Send community scores to neighbors
          triplet.sendToSrc(triplet.dstAttr)
          triplet.sendToDst(triplet.srcAttr)
        },
        (a, b) => a.zip(b).map { case (x, y) => x + y } // Combine incoming scores
      )

      // Update memberships with gradient ascent
      memberships = memberships.innerJoin(incomingMessages) { (_, current, incoming) =>
        current.zip(incoming).map { case (c, i) =>
          c + learningRate * (i - c) // Simplified update rule
        }
      }
    }

    // Assign nodes to communities
    val communities = memberships.mapValues(scores =>
      scores.zipWithIndex.maxBy(_._1)._2 // Assign to community with max score
    )

    println("Detected Communities:")
    communities.collect().foreach(println)
  }
}
