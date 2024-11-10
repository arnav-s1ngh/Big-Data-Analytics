from pyspark import SparkContext
from pyspark.sql import SparkSession
from neo4j import GraphDatabase
import networkx as nx
from collections import defaultdict
import numpy as np

# Initialize Spark session

url = "neo4j://localhost:7687"
username = "neo4j"
password = "12345678"
dbname = "bdaassignment3"


import findspark
findspark.init()


spark=SparkSession.builder.config("spark.driver.host", "localhost").appName('appname').getOrCreate()


# Neo4j connection configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # Replace with actual password


def get_citation_graph():
    """Retrieve citation graph from Neo4j and convert to NetworkX format"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Get all nodes
        result = session.run("""
            MATCH (p:Paper)
            RETURN p.id AS paper_id
        """)
        nodes = [record["paper_id"] for record in result]

        # Get all citation relationships
        result = session.run("""
            MATCH (p1:Paper)-[r:CITES]->(p2:Paper)
            RETURN p1.id AS source, p2.id AS target
        """)
        edges = [(record["source"], record["target"]) for record in result]

    driver.close()

    # Create NetworkX directed graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G


def parallel_simrank(graph, query_nodes, c, max_iter=100, tolerance=1e-4):
    """
    Compute SimRank similarities using Spark for parallelization
    Args:
        graph: NetworkX graph
        query_nodes: List of nodes to compute similarities for
        c: Decay factor
        max_iter: Maximum number of iterations
        tolerance: Convergence tolerance
    """
    # Convert graph to adjacency matrix format for easier processing
    adj_matrix = nx.to_scipy_sparse_array(graph)
    n = adj_matrix.shape[0]

    # Create node index mapping
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    # Initialize similarity matrix
    sim_matrix = np.identity(n)

    # Create RDD of node pairs for parallel processing
    node_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    node_pairs_rdd = spark.sparkContext.parallelize(node_pairs)

    def update_similarity(pair):
        i, j = pair
        if i == j:
            return (i, j, 1.0)

        in_neighbors_i = set(graph.predecessors(idx_to_node[i]))
        in_neighbors_j = set(graph.predecessors(idx_to_node[j]))

        if not in_neighbors_i or not in_neighbors_j:
            return (i, j, 0.0)

        sum_sim = 0.0
        for u in in_neighbors_i:
            u_idx = node_to_idx[u]
            for v in in_neighbors_j:
                v_idx = node_to_idx[v]
                sum_sim += sim_matrix[u_idx][v_idx]

        new_sim = (c / (len(in_neighbors_i) * len(in_neighbors_j))) * sum_sim
        return (i, j, new_sim)

    # Run iterations
    for _ in range(max_iter):
        new_sims = node_pairs_rdd.map(update_similarity).collect()

        # Update similarity matrix
        max_diff = 0.0
        for i, j, sim in new_sims:
            diff = abs(sim_matrix[i][j] - sim)
            max_diff = max(max_diff, diff)
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

        if max_diff < tolerance:
            break

    # Get similarities for query nodes
    results = {}
    for query_node in query_nodes:
        query_idx = node_to_idx[query_node]
        node_sims = [(idx_to_node[i], sim_matrix[query_idx][i])
                     for i in range(n) if i != query_idx]
        results[query_node] = sorted(node_sims, key=lambda x: x[1], reverse=True)[:10]

    return results


def main():
    # Get citation graph from Neo4j
    graph = get_citation_graph()

    # Query nodes
    query_nodes = [2982615777, 1556418098]

    # Run SimRank with different C values
    c_values = [0.7, 0.8, 0.9]

    for c in c_values:
        print(f"\nRunning SimRank with C = {c}")
        similarities = parallel_simrank(graph, query_nodes, c)

        for query_node in query_nodes:
            print(f"\nTop 10 similar papers to {query_node}:")
            for similar_node, similarity in similarities[query_node]:
                print(f"Node: {similar_node}, Similarity: {similarity:.4f}")


main()
spark.stop()
