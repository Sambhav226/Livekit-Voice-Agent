import time
import traceback
import os
import httpx
from dotenv import load_dotenv
from typing import List, Dict
from loguru import logger

# Use REST API instead of gRPC to avoid DNS issues
from pinecone import Pinecone

from document_schema import Document

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

COHERE_EMBEDDING_BASE_URL = "https://api.cohere.com/v2/embed"
DIMENSION = 1024

class VectorSearch:
    def __init__(self, index_name: str):
        self.index_name = index_name
        
        # Use REST API instead of gRPC
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(name=self.index_name)
        
        self.bm25 = None 

        self.cohere_headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Log initialization
        logger.info(f"VectorSearch initialized with index: {self.index_name}")
        

    async def create_embeddings(self, texts: List[str]) -> List[Dict]:
        start_time = time.time()
        logger.debug(f"Creating embeddings for {len(texts)} texts")
        
        payload = {
            "model": "embed-v4.0",
            "texts": texts,
            "input_type": "search_query",
            "embedding_types": ["float"],
            "output_dimension": DIMENSION,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    COHERE_EMBEDDING_BASE_URL,
                    headers=self.cohere_headers,
                    json=payload
                )

            if response.status_code != 200:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_details = response.json()
                    logger.error(f"Error Response: {error_details}")
                    error_msg += f" - {error_details.get('error', {}).get('message', 'Unknown error')}"
                except:
                    pass
                raise Exception(error_msg)

            embeddings_by_type = response.json().get("embeddings", {})
            embeddings = embeddings_by_type.get("float", [])
            
            logger.debug(f"Generated {len(embeddings)} embeddings in {time.time() - start_time:.3f}s")
            if embeddings:
                logger.debug(f"First embedding dimension: {len(embeddings[0])}")
            
            return [{"embedding": emb} for emb in embeddings]

        except httpx.HTTPError as e:
            logger.error(f"Embedding HTTP error: {str(e)}")
            raise Exception(f"Embedding HTTP error: {str(e)}\n{traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise Exception(f"Embedding error: {str(e)}\n{traceback.format_exc()}")

    async def rerank(self, docs: List[Document], query: str, topn: int = 5) -> List[Dict]:
        rerank_url = "https://api.cohere.com/v2/rerank"
        doc_texts = [
            doc.text for doc in docs
            if isinstance(doc.text, str) and doc.text.strip()
        ]
        
        logger.debug(f"Reranking {len(doc_texts)} documents")
        
        if not doc_texts:
            logger.warning("No valid document texts for reranking.")
            return []

        payload = {
            "model": "rerank-v3.5",
            "query": query,
            "top_n": topn,
            "documents": doc_texts,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    rerank_url,
                    headers=self.cohere_headers,
                    json=payload,
                )

            if response.status_code != 200:
                logger.error(f"Rerank API error: {response.status_code}")
                return [{"document": doc, "relevance_score": 1.0} for doc in docs]

            reranked = []
            results = response.json().get("results", [])
            logger.debug(f"Rerank returned {len(results)} results")
            
            for r in results:
                idx = r.get("index")
                score = r.get("relevance_score", 1.0)
                logger.debug(f"Rerank result - index: {idx}, score: {score}")
                
                if isinstance(idx, int) and 0 <= idx < len(docs):
                    reranked.append({
                        "document": docs[idx],
                        "relevance_score": score
                    })

            return reranked
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Return original docs with default score if reranking fails
            return [{"document": doc, "relevance_score": 1.0} for doc in docs]

    def hybrid_score_norm(self, dense, sparse, alpha=0.75):
        return dense, {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }

    def query(self, dense_vec, sparse_vec, namespace, category, topk=5):
        logger.debug(f"Querying Pinecone - namespace: {namespace}, category: {category}, topk: {topk}")
        logger.debug(f"Dense vector length: {len(dense_vec) if dense_vec else 'None'}")
        logger.debug(f"Sparse vector: {sparse_vec}")
        
        try:
            hdense, hsparse = self.hybrid_score_norm(dense_vec, sparse_vec)
            
            # Build query parameters
            query_params = {
                "vector": hdense,
                "top_k": topk,
                "include_metadata": True,
                "include_values": False,
            }
            
            # Add namespace if provided
            if namespace and namespace.strip():
                query_params["namespace"] = namespace
                logger.debug(f"Using namespace: {namespace}")
            else:
                logger.debug("No namespace specified, querying default namespace")
            
            # Add sparse vector if it has values
            if hsparse["indices"] and hsparse["values"]:
                query_params["sparse_vector"] = hsparse
                logger.debug("Using hybrid query with sparse vector")
            else:
                logger.debug("Using dense-only query")
            
            # Add filter if category is specified
            if category and category.strip():
                query_params["filter"] = {"doc_category": {"$eq": category}}
                logger.debug(f"Using category filter: {category}")
            else:
                logger.debug("No category filter applied")
            
            # Execute query with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Query attempt {attempt + 1}/{max_retries}")
                    result = self.index.query(**query_params)
                    
                    logger.debug(f"Pinecone query returned {len(result.get('matches', []))} matches")
                    for i, match in enumerate(result.get('matches', [])[:3]):  # Log first 3 matches
                        logger.debug(f"Match {i}: id={match.get('id')}, score={match.get('score')}, metadata_keys={list(match.get('metadata', {}).keys())}")
                    
                    return result
                    
                except Exception as retry_error:
                    logger.warning(f"Query attempt {attempt + 1} failed: {retry_error}")
                    if attempt == max_retries - 1:
                        raise retry_error
                    time.sleep(1)  # Wait before retry
            
        except Exception as e:
            logger.error(f"Pinecone query failed after all retries: {str(e)}")
            logger.error(f"Query parameters were: namespace={namespace}, category={category}, topk={topk}")
            logger.error(f"Full error: {traceback.format_exc()}")
            return {"matches": []}

    async def retrieval(self, query: str, namespace: str, doc_category: str, topn=5, rerank_threshold=0.1):
        logger.info(f"[RETRIEVAL] Started for query: '{query}' | namespace: '{namespace}' | category: '{doc_category}'")
        
        # Validate inputs
        if not query or not query.strip():
            logger.error("Empty query provided")
            return []
        
        try:
            # Step 1: Generate dense embedding
            logger.debug("Step 1: Generating embeddings")
            emb_data = await self.create_embeddings([query])
            
            if not emb_data or not emb_data[0].get("embedding"):
                logger.error("Failed to generate embeddings")
                return []
            
            emb_vec = emb_data[0]["embedding"]
            logger.debug(f"Generated embedding with dimension: {len(emb_vec)}")

            # Step 2: Create empty sparse vector
            sparse_vec = {"indices": [], "values": []}

            # Step 3: Query Pinecone
            logger.debug("Step 3: Querying Pinecone")
            result = self.query(emb_vec, sparse_vec, namespace, doc_category, topk=topn)

            if not result or "matches" not in result or not result["matches"]:
                logger.warning("[RETRIEVAL] No matches found from Pinecone query")
                
                # Try without filters to diagnose
                logger.debug("Trying query without filters for diagnosis...")
                try:
                    fallback_result = self.query(emb_vec, sparse_vec, "", "", topk=topn)
                    if fallback_result and fallback_result.get("matches"):
                        logger.info(f"Found {len(fallback_result['matches'])} matches without filters")
                        logger.info("This suggests the issue is with namespace or category filtering")
                    else:
                        logger.warning("No matches found even without filters - index might be empty")
                except Exception as fallback_error:
                    logger.error(f"Fallback query also failed: {fallback_error}")
                
                return []

            # Step 4: Process matches into Document objects
            logger.debug("Step 4: Processing matches into Document objects")
            docs = []
            for i, m in enumerate(result["matches"]):
                logger.debug(f"Processing match {i}: {m.get('id')}")
                
                metadata = m.get("metadata", {})
                text = metadata.get("text", "")

                if not isinstance(text, str) or not text.strip():
                    logger.warning(f"Skipping doc {m.get('id')} due to missing/invalid text")
                    continue

                try:
                    # Create a copy of metadata without 'text' to avoid conflicts
                    doc_metadata = {k: v for k, v in metadata.items() if k != "text"}
                    
                    # Create Document with explicit parameters
                    doc = Document(
                        id=m.get("id"),
                        text=text,
                        query=query,
                        **doc_metadata  # Pass remaining metadata as kwargs
                    )
                    docs.append(doc)
                    logger.debug(f"Successfully created Document for {m.get('id')}")
                except Exception as e:
                    logger.warning(f"Failed to construct Document for {m.get('id')}: {e}")
                    logger.debug(f"Metadata keys: {list(metadata.keys())}")

            if not docs:
                logger.warning("[RETRIEVAL] No valid documents created from matches")
                return []

            logger.debug(f"Created {len(docs)} valid documents")

            # Step 5: Rerank with Cohere
            logger.debug("Step 5: Reranking documents")
            reranked = await self.rerank(docs, query, topn=topn)

            if not reranked:
                logger.warning("Reranking returned no results")
                return []

            # Step 6: Filter by relevance threshold
            logger.debug(f"Step 6: Filtering by threshold {rerank_threshold}")
            filtered = []
            for item in reranked:
                score = item.get("relevance_score", 0.2)
                if score >= rerank_threshold:
                    filtered.append(item["document"])
                    logger.debug(f"Document passed threshold: score={score}")
                else:
                    logger.debug(f"Document filtered out: score={score} < {rerank_threshold}")

            logger.info(f"[RETRIEVAL] Returning {len(filtered)} documents after reranking and filtering")
            return filtered

        except Exception as e:
            logger.error(f"[RETRIEVAL] Failed with error: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []