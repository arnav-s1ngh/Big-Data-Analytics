CREATE INDEX FOR (p:Paper) ON (p.id);
CALL apoc.periodic.iterate(
  "CALL apoc.load.json('file:///train.json') YIELD value RETURN value",  // Load JSON in batches
  "
  UNWIND value AS row
  MERGE (p:Paper {id: row.paper})  // Create or match citing paper node

  FOREACH (refId IN row.reference |
      MERGE (ref:Paper {id: refId})  // Create or match referenced paper node
      MERGE (p)-[:cite]->(ref)      // Create citation relationship
  )",
  {batchSize: 1, parallel: true}  // Batch size and parallel execution
)
YIELD batches, total
RETURN batches, total;
