from pyspark.sql import SparkSession

spark=(
    SparkSession.builder.config("neo4j.url", "neo4j://localhost:7687")
    .appName("BDA Assignment 3")
    .config("spark.jars.packages","org.neo4j:neo4j-connector-apache-spark_2.12:5.3.2_for_spark_3")
    .config("neo4j.authentication.basic.username", "neo4j")
    .config("neo4j.authentication.basic.password", "12345678")
    .config("neo4j.database", "assignmentbda3")
    .getOrCreate()
)

node_df=(
    spark.read.format("org.neo4j.spark.DataSource")
    .option("labels","Paper")
    .load()
)
query = """
MATCH (p1:Paper)-[:cite]->(p2:Paper)
RETURN p1.id AS citing_paper, p2.id AS cited_paper
"""

# Run the query on Neo4j and load the result into a DataFrame
edge_df = (
    spark.read.format("org.neo4j.spark.DataSource")
    .option("query", query)
    .load()
    .show()
)
