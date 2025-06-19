# app.py
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from dotenv import load_dotenv
from PIL import Image
import io
import sqlitecloud

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure correct path to db on Vercel (read-only file access)
SQLITECLOUD_URL = os.getenv("SQLITECLOUD_URL")

SIMILARITY_THRESHOLDS = [0.68, 0.58, 0.48]
MAX_RESULTS = 10  # Increased to get more context
load_dotenv()
MAX_CONTEXT_CHUNKS = 4  # Increased number of chunks per source
API_KEY = os.getenv("API_KEY")
N_API_KEY = os.getenv("N_API_KEY")
print("Loaded API_KEY (partial):", API_KEY[:10])

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key is set
if not API_KEY:
    logger.error("API_KEY environment variable is not set. The application will not function correctly.")

def get_sqlitecloud_connection():
    return sqlitecloud.connect(SQLITECLOUD_URL)
    
# Create a connection to the SQLite database
def get_db_connection():
    try:
        return get_sqlitecloud_connection()
    except Exception as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Vector similarity calculation with improved handling
def cosine_similarity(vec1, vec2):
    try:
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return 0 similarity on error rather than crashing

def detect_image_mime_type(base64_str: str) -> str:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        format = image.format.lower()
        mime = f"image/{format}" if format != "jpeg" else "image/jpeg"
        logger.info(f"Detected image format: {format}, MIME type: {mime}")
        return mime
    except Exception as e:
        logger.error(f"Error detecting image format: {e}")
        return "image/jpeg"  # Default fallback

# Function to get embedding from aipipe proxy with retry mechanism
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Getting embedding for text (length: {len(text)})")
            # Call the embedding API through aipipe proxy
            url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            logger.info("Sending request to embedding API")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received embedding")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error getting embedding (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)  # Wait before retry

# Function to find similar content in the database with improved logic
async def find_similar_content(query_embedding, conn, threshold):
    try:
        logger.info("Finding similar content in database")
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
        logger.info("Querying discourse chunks")
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
               likes, chunk_index, content, url, embedding 
        FROM discourse_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        discourse_chunks = cursor.fetchall()
        logger.info(f"Processing {len(discourse_chunks)} discourse chunks")
        processed_count = 0
        
        for chunk in discourse_chunks:
            try:
                embedding = json.loads(chunk[11])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= threshold:
                    # Ensure URL is properly formatted
                    url = chunk[10]
                    if not url.startswith("http"):
                        # Fix missing protocol
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                    
                    results.append({
                        "source": "discourse",
                        "id": chunk[0],
                        "post_id": chunk[1],
                        "topic_id": chunk[2],
                        "title": chunk[3],
                        "url": chunk[10],
                        "content": chunk[9],
                        "author": chunk[5],
                        "created_at": chunk[6],
                        "chunk_index": chunk[8],
                        "similarity": float(similarity)
                    })

                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(discourse_chunks)} discourse chunks")
                    
            except Exception as e:
                logger.error(f"Error processing discourse chunk {chunk[0]}: {e}")
        
        # Search markdown chunks
        logger.info("Querying markdown chunks")
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
        FROM markdown_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        markdown_chunks = cursor.fetchall()
        logger.info(f"Processing {len(markdown_chunks)} markdown chunks")
        processed_count = 0
        
        for chunk in markdown_chunks:
            try:
                embedding = embedding = json.loads(chunk[6])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= threshold:
                    # Ensure URL is properly formatted
                    url = chunk[2]
                    if not url or not url.startswith("http"):
                        # Use a default URL if missing
                        url = f"https://docs.onlinedegree.iitm.ac.in/{chunk[1]}"
                    
                    results.append({
                        "source": "markdown",
                        "id": chunk[0],
                        "title": chunk[1],
                        "url": chunk[2],                 
                        "content": chunk[5],
                        "chunk_index": chunk[4],
                        "similarity": float(similarity)
                    })

                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count}/{len(markdown_chunks)} markdown chunks")
                    
            except Exception as e:
                logger.error(f"Error processing markdown chunk {chunk[0]}: {e}")
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(results)} relevant results above threshold")
        
        # Group by source document and keep most relevant chunks
        grouped_results = {}
        
        for result in results:
            # Create a unique key for the document/post
            if result["source"] == "discourse":
                key = f"discourse_{result['post_id']}"
            else:
                key = f"markdown_{result['title']}"
            
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].append(result)
        
        # For each source, keep only the most relevant chunks
        final_results = []
        for key, chunks in grouped_results.items():
            # Sort chunks by similarity
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            # Keep top chunks
            final_results.extend(chunks[:MAX_CONTEXT_CHUNKS])
        
        # Sort again by similarity
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top results, limited by MAX_RESULTS
        logger.info(f"Returning {len(final_results[:MAX_RESULTS])} final results after grouping")
        return final_results[:MAX_RESULTS]
    except Exception as e:
        error_msg = f"Error in find_similar_content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise

async def enrich_with_adjacent_chunks(conn, relevant_results):
    logger.info(f"Enriching {len(relevant_results)} results with adjacent chunks (based on id)")

    enriched_results = []
    cursor = conn.cursor()

    for result in relevant_results:
        try:
            base_id = result["id"]
            additional_content = result["content"]

            # Add previous chunk (id - 1)
            cursor.execute("SELECT content FROM discourse_chunks WHERE id = ?", (base_id - 1,))
            prev_chunk = cursor.fetchone()
            if prev_chunk:
                additional_content = prev_chunk[0] + " " + additional_content

            # Add next chunk (id + 1)
            cursor.execute("SELECT content FROM discourse_chunks WHERE id = ?", (base_id + 1,))
            next_chunk = cursor.fetchone()
            if next_chunk:
                additional_content += " " + next_chunk[0]

            # Final enriched result
            enriched_results.append({
                **result,
                "content": additional_content
            })

        except Exception as e:
            logger.error(f"Error enriching result {result['id']}: {e}")
            enriched_results.append(result)  # fallback to original

    logger.info(f"Successfully enriched {len(enriched_results)} results")
    return enriched_results


# Function to generate an answer using LLM with improved prompt
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:    
        try:
            logger.info(f"Generating answer for question: '{question[:50]}...'")
            context = ""
            for result in relevant_results:
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                context += f"\n\n{source_type} (URL: {result['url']}):\n{result['content'][:1500]}"
            
            # Prepare improved prompt
            prompt = f"""Answer the following question based ONLY on the provided context. 
            If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Return your response in this exact format:
            1. A comprehensive yet concise answer
            2. A "Sources:" section that lists the URLs and relevant text snippets you used to answer
            
            Sources must be in this exact format:
            Sources:
            1. URL: [exact_url_1], Text: [brief quote or description]
            2. URL: [exact_url_2], Text: [brief quote or description]
            
            Make sure the URLs are copied exactly from the context without any changes.
            """
            
            logger.info("Sending request to LLM API")
            url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3  # Lower temperature for more deterministic outputs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(3 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception generating answer: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)  # Wait before retry

# Function to process multimodal content (text + image)
async def process_multimodal_query(question, image_base64):
    if not N_API_KEY:
        error_msg = "API_KEY environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
    try:
        logger.info(f"Processing query: '{question[:50]}...', image provided: {image_base64 is not None}")
        if not image_base64:
            logger.info("No image provided, processing as text-only query")
            return await get_embedding(question)
        
        logger.info("Processing multimodal query with image")
        # Call the GPT-4o Vision API to process the image and question
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Authorization": N_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Format the image for the API
        mime_type = detect_image_mime_type(image_base64)
        image_content = f"data:{mime_type};base64,{image_base64}"

        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Look at this image and tell me what you see related to this question: {question}"},
                        {"type": "image_url", "image_url": {"url": image_content}}
                    ]
                }
            ]
        }
        
        logger.info("Sending request to Vision API")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    logger.info(f"Received image description: '{image_description[:50]}...'")
                    
                    # Combine the original question with the image description
                    combined_query = f"{question}\nImage context: {image_description}"
                    
                    # Get embedding for the combined query
                    return await get_embedding(combined_query)
                else:
                    error_text = await response.text()
                    logger.error(f"Error processing image (status {response.status}): {error_text}")
                    # Fall back to text-only query
                    logger.info("Falling back to text-only query")
                    return await get_embedding(question)
    except Exception as e:
        logger.error(f"Exception processing multimodal query: {e}")
        logger.error(traceback.format_exc())
        # Fall back to text-only query
        logger.info("Falling back to text-only query due to exception")
        return await get_embedding(question)

# Function to parse LLM response and extract answer and sources with improved reliability
def parse_llm_response(response):
    try:
        logger.info("Parsing LLM response")
        
        # First try to split by "Sources:" heading
        parts = response.split("Sources:", 1)
        
        # If that doesn't work, try alternative formats
        if len(parts) == 1:
            # Try other possible headings
            for heading in ["Source:", "References:", "Reference:"]:
                if heading in response:
                    parts = response.split(heading, 1)
                    break
        
        answer = parts[0].strip()
        links = []
        
        if len(parts) > 1:
            sources_text = parts[1].strip()
            source_lines = sources_text.split("\n")
            
            for line in source_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Remove list markers (1., 2., -, etc.)
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)
                
                # Extract URL and text using more flexible patterns
                url_match = re.search(r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|\[(http[^\]]+)\]|URL:\s*(http\S+)|url:\s*(http\S+)|(http\S+)', line, re.IGNORECASE)
                text_match = re.search(r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|[""](.*?)[""]|Text:\s*"(.*?)"|text:\s*"(.*?)"', line, re.IGNORECASE)
                
                if url_match:
                    # Find the first non-None group from the regex match
                    url = next((g for g in url_match.groups() if g), "")
                    url = url.strip()
                    
                    # Default text if no match
                    text = "Source reference"
                    
                    # If we found a text match, use it
                    if text_match:
                        # Find the first non-None group from the regex match
                        text_value = next((g for g in text_match.groups() if g), "")
                        if text_value:
                            text = text_value.strip()
                    
                    # Only add if we have a valid URL
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})
        
        logger.info(f"Parsed answer (length: {len(answer)}) and {len(links)} sources")
        return {"answer": answer, "links": links}
    except Exception as e:
        error_msg = f"Error parsing LLM response: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        # Return a basic response structure with the error
        return {
            "answer": "Error parsing the response from the language model.",
            "links": []
        }

# Define API routes
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        # Log the incoming request
        logger.info(f"Received query request: question='{request.question[:50]}...', image_provided={request.image is not None}")
        
        if not API_KEY:
            error_msg = "API_KEY environment variable not set"
            logger.error(error_msg)
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
            
        conn = get_db_connection()
        
        try:
            # Process the query (handle text and optional image)
            logger.info("Processing query and generating embedding")
            query_embedding = await process_multimodal_query(
                request.question,
                request.image
            )
            
            # Find similar content
            logger.info("Finding similar content")
            relevant_results = []
            used_similarity_threshold = None
            
            for threshold in SIMILARITY_THRESHOLDS:
                logger.info(f"Trying similarity threshold: {threshold}")
                results = await find_similar_content(query_embedding, conn, threshold)
                
                if results:
                    relevant_results = results
                    used_similarity_threshold = threshold
                    break
            
            if not relevant_results:
                logger.info("No relevant results found after all threshold attempts")
                return {
                    "answer": "I couldn't find any relevant information in my knowledge base.",
                    "links": []
                }

            
            # Enrich results with adjacent chunks for better context
            logger.info("Enriching results with adjacent chunks")
            enriched_results = await enrich_with_adjacent_chunks(conn, relevant_results)
            
            # Generate answer
            logger.info("Generating answer")
            llm_response = await generate_answer(request.question, enriched_results)
            
            if isinstance(llm_response, dict):
                answer = llm_response.get("answer", "")
                sources = llm_response.get("links", [])
            else:
                # If it’s a plain string response from LLM (fallback or unexpected format)
                answer = str(llm_response)
                sources = []
            
            # Fallback if LLM fails and similarity threshold was high
            if (
                answer.strip().lower() == "i don't have enough information to answer this question." and 
                used_similarity_threshold == 0.68 and 
                len(relevant_results) > 0 and 
                relevant_results[0]["source"] == "discourse"  # ✅ Only do this for discourse chunks
            ):
                logger.info("⚠️ LLM did not answer even though similarity ≥ 0.68. Trying next 3 rows based on ID (discourse only)")
            
                try:
                    base_id = relevant_results[0]["id"]
                    cursor = conn.cursor()
            
                    # Get next 3 discourse_chunks by id (if available)
                    cursor.execute("""
                        SELECT content FROM discourse_chunks
                        WHERE id > ?
                        ORDER BY id ASC
                        LIMIT 3
                    """, (base_id,))
                    next_chunks = cursor.fetchall()
            
                    if next_chunks:
                        extra_context = " ".join(chunk[0] for chunk in next_chunks if chunk and chunk[0])
                        enriched_results[0]["content"] += " " + extra_context
            
                        logger.info("🔁 Re-generating LLM answer with additional adjacent context")
                        llm_response = await generate_answer(request.question, enriched_results)
            
                        if isinstance(llm_response, dict):
                            answer = llm_response.get("answer", "")
                            sources = llm_response.get("links", [])
                        else:
                            answer = str(llm_response)
                            sources = []
            
                except Exception as e:
                    logger.error(f"❌ Error in fallback enrichment with next 3 rows: {e}")
            
            # Parse the response
            logger.info("Parsing LLM response")
            result = parse_llm_response(llm_response)
            
            # If links extraction failed, create them from the relevant results
            if not result["links"]:
                logger.info("No links extracted, creating from relevant results")
                # Create a dict to deduplicate links from the same source
                links = []
                unique_urls = set()
                
                for res in relevant_results[:5]:  # Use top 5 results
                    url = res["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                        links.append({"url": url, "text": snippet})
                
                result["links"] = links
            
            # Log the final result structure (without full content for brevity)
            logger.info(f"Returning result: answer_length={len(result['answer'])}, num_links={len(result['links'])}")
            
            # Return the response in the exact format required
            return result
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
        finally:
            conn.close()
    except Exception as e:
        # Catch any exceptions at the top level
        error_msg = f"Unhandled exception in query_knowledge_base: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Try to connect to the database as part of health check
        conn = get_sqlitecloud_connection()
        cursor = conn.cursor()
        
        # Check if tables exist and have data
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        
        # Check if any embeddings exist
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 
    