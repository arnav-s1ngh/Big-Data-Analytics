import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object AGMCommunityDetection {

  def main(args: Array[String]): Unit={
    val spark=SparkSession.builder().appName("AGM Community Detection").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    // Nodes
    val nodes="/Users/arnavsingh/Downloads/git_web_ml/musae_git_target.csv"
    val nodesRDD=spark.read.option("header","true").csv(nodes).rdd.map(row=>(row.getString(0).toLong,row.getString(1)))
    // Edges
    val edges="/Users/arnavsingh/Downloads/git_web_ml/musae_git_edges.csv"
    val edgesRDD=spark.read.option("header","true").csv(edges).rdd.flatMap(row=>Seq(Edge(row.getString(0).toLong,row.getString(1).toLong,1),Edge(row.getString(1).toLong,row.getString(0).toLong,1)))
    // Graph
    val graph=Graph(nodesRDD,edgesRDD)
    println(s"Number of Vertices:- ${graph.vertices.count()}\nNumber of Edges:- ${graph.edges.count()} edges.")
  }
}
