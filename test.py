import sqlite3
import json
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
import logging

# === Setup Logging and API ===
load_dotenv()
API_KEY = os.getenv("N_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "knowledge_base.db"
ROW_ID_TO_UPDATE = 4073

# === Embedding Request ===
async def get_embedding(text):
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return json.dumps(result["data"][0]["embedding"]).encode()
            else:
                raise Exception(await response.text())

# === Main Update Logic ===
async def update_embedding():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Step 1: Fetch the row to update
    cursor.execute("SELECT content FROM discourse_chunks WHERE id = ?", (ROW_ID_TO_UPDATE,))
    row = cursor.fetchone()

    if not row:
        logger.error(f"No row found with id={ROW_ID_TO_UPDATE}")
        return

    content = row[0]
    logger.info(f"📄 Generating embedding for ID {ROW_ID_TO_UPDATE} (content length: {len(content)})")

    # Step 2: Generate new embedding
    embedding_blob = await get_embedding(content)
    logger.info("✅ Embedding generated.")

    # Step 3: Update the embedding field
    cursor.execute("UPDATE discourse_chunks SET embedding = ? WHERE id = ?", (embedding_blob, ROW_ID_TO_UPDATE))
    conn.commit()
    conn.close()
    logger.info(f"📝 Updated embedding for row ID {ROW_ID_TO_UPDATE}")

# === Run the Script ===
asyncio.run(update_embedding())
