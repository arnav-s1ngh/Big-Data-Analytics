CREATE INDEX FOR (p:Paper) ON (p.id);
CALL apoc.periodic.iterate(
  "CALL apoc.load.json('file:///train.json') YIELD value RETURN value",
  "
  UNWIND value AS row
  MERGE (p:Paper {id: row.paper})  
  FOREACH (refId IN row.reference |
      MERGE (ref:Paper {id: refId})
      MERGE (p)-[:cite]->(ref)    
  )",
  {batchSize: 1, parallel: true}  
)
YIELD batches, total
RETURN batches, total;
