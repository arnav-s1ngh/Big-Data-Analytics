from neo4j import GraphDatabase
from pyspark.sql import SparkSession
import networkx as nx
import numpy as np
from tqdm import tqdm
import gc


class SimRankAnalyzer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="12345678", database="neo4j",
                 batch_size=1000):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.batch_size = batch_size
        self.spark = SparkSession.builder \
            .appName("SimRank Analysis") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .getOrCreate()

    def close(self):
        self.driver.close()
        self.spark.stop()

    def get_nodes_batch(self, skip=0):
        """Retrieve nodes in batches"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p:Paper)
                RETURN p.id as node_id
                SKIP $skip
                LIMIT $limit
                """,
                                 skip=skip,
                                 limit=self.batch_size
                                 )
            return [record["node_id"] for record in result]

    def get_edges_batch(self, skip=0):
        """Retrieve edges in batches"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (p1:Paper)-[r:CITES]->(p2:Paper)
                RETURN p1.id as source, p2.id as target
                SKIP $skip
                LIMIT $limit
                """,
                                 skip=skip,
                                 limit=self.batch_size
                                 )
            return [(record["source"], record["target"]) for record in result]

    def get_citation_graph(self):
        """Retrieve citation graph from Neo4j in batches and convert to NetworkX format"""
        G = nx.DiGraph()

        # Get nodes in batches
        print("Fetching nodes in batches...")
        skip = 0
        while True:
            nodes_batch = self.get_nodes_batch(skip)
            if not nodes_batch:
                break
            G.add_nodes_from(nodes_batch)
            skip += self.batch_size
            print(f"Loaded {G.number_of_nodes()} nodes so far...")

        # Get edges in batches
        print("\nFetching edges in batches...")
        skip = 0
        while True:
            edges_batch = self.get_edges_batch(skip)
            if not edges_batch:
                break
            G.add_edges_from(edges_batch)
            skip += self.batch_size
            print(f"Loaded {G.number_of_edges()} edges so far...")

        print(f"\nFinal graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def compute_simrank_batch(self, G, nodes_batch, c=0.8, max_iter=10):
        """Compute SimRank scores for a batch of nodes"""
        similarities = {}

        # Pre-compute predecessors for the entire graph (this is used frequently)
        pred_dict = {node: list(G.predecessors(node)) for node in G.nodes()}

        for query_node in nodes_batch:
            print(f"Computing similarities for query node: {query_node}")
            # Initialize similarity scores for this query node
            sim_dict = {node: 0.0 for node in G.nodes()}
            sim_dict[query_node] = 1.0

            # Iterate max_iter times
            for iter_num in range(max_iter):
                print(f"Iteration {iter_num + 1}/{max_iter}")
                new_sim = {node: 0.0 for node in G.nodes()}
                new_sim[query_node] = 1.0

                # Process nodes in batches to compute similarities
                node_list = list(G.nodes())
                for i in range(0, len(node_list), self.batch_size):
                    node_batch = node_list[i:i + self.batch_size]
                    for target_node in node_batch:
                        if target_node == query_node:
                            continue

                        pred_query = pred_dict[query_node]
                        pred_target = pred_dict[target_node]

                        if not pred_query or not pred_target:
                            continue

                        total = 0.0
                        # Process predecessor pairs in batches
                        pred_batch_size = 100  # Smaller batch size for predecessor pairs
                        for i in range(0, len(pred_query), pred_batch_size):
                            pred_q_batch = pred_query[i:i + pred_batch_size]
                            for j in range(0, len(pred_target), pred_batch_size):
                                pred_t_batch = pred_target[j:j + pred_batch_size]
                                for u in pred_q_batch:
                                    for v in pred_t_batch:
                                        total += sim_dict[u] * sim_dict[v]

                        if len(pred_query) > 0 and len(pred_target) > 0:
                            new_sim[target_node] = (c * total) / (len(pred_query) * len(pred_target))

                sim_dict = new_sim
                gc.collect()  # Force garbage collection after each iteration

            similarities[query_node] = sim_dict
            gc.collect()  # Force garbage collection after processing each node

        return similarities

    def get_top_k_similar(self, similarities_dict, k=10):
        """Get top-k similar nodes from similarity dictionary"""
        sorted_nodes = sorted(similarities_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:k]

    def analyze_similarities(self, query_nodes, c_values, top_k=10):
        """Analyze similarities for given query nodes and c values"""
        print("Loading graph...")
        G = self.get_citation_graph()
        results = {}

        # Process query nodes in batches
        batch_size = 2  # Small batch size since we have few query nodes
        for c in c_values:
            print(f"\nComputing SimRank with c = {c}")
            results[c] = {}

            for i in range(0, len(query_nodes), batch_size):
                batch = query_nodes[i:i + batch_size]
                print(f"\nProcessing batch of {len(batch)} query nodes...")

                similarities = self.compute_simrank_batch(G, batch, c=c)

                for query_node in batch:
                    top_similar = self.get_top_k_similar(similarities[query_node], k=top_k)
                    results[c][query_node] = top_similar

                    print(f"\nTop {top_k} similar nodes for query node {query_node} (c={c}):")
                    for node, score in top_similar:
                        print(f"Node: {node}, Similarity: {score:.4f}")

                gc.collect()

        # Save results to a file
        with open('simrank_results.txt', 'w') as f:
            f.write("SimRank Analysis Results\n")
            f.write("=======================\n\n")
            for c in c_values:
                f.write(f"\nResults for c = {c}\n")
                f.write("--------------------\n")
                for query_node in query_nodes:
                    f.write(f"\nQuery Node: {query_node}\n")
                    for node, score in results[c][query_node]:
                        f.write(f"Similar Node: {node}, Score: {score:.4f}\n")

        return results


def main():
    # Initialize analyzer with provided credentials
    analyzer = SimRankAnalyzer(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="12345678",
        database="neo4j",
        batch_size=1000  # Adjust if needed based on memory
    )

    # Query nodes and c values as specified in the assignment
    query_nodes = [2982615777, 1556418098]
    c_values = [0.7, 0.8, 0.9]

    try:
        # Run analysis
        print("Starting SimRank analysis...")
        results = analyzer.analyze_similarities(query_nodes, c_values)
        print("\nAnalysis complete! Results have been saved to 'simrank_results.txt'")

    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
