import ast
import zipfile
import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
import requests
from scipy.spatial.distance import cosine
import logging
from typing import Dict, Any, Optional
import tempfile
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Assignment Checker")

class EmbeddingModel(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

# Load configuration from environment variables
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/embeddings")
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))

# MongoDB setup with provided URI
try:
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB_NAME", "assignment_checker")
    collection_name = os.getenv("MONGODB_COLLECTION_NAME", "embeddings")
    
    if not uri:
        raise ValueError("MongoDB URI not found in environment variables")
    
    mongo_client = MongoClient(uri)
    db = mongo_client[db_name]
    collection = db[collection_name]
    # Test connection
    mongo_client.server_info()
    logger.info("Successfully connected to MongoDB Atlas")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

def extract_functions_from_file(file_path: str) -> Dict[str, str]:
    """Extract function definitions from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        tree = ast.parse(code)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        return {func.name: ast.get_source_segment(code, func) for func in functions}
    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {e}")
        return {}

def extract_functions_from_zip(zip_path: str) -> Dict[str, str]:
    """Extract all functions from Python files in a zip."""
    func_codes = {}
    with tempfile.TemporaryDirectory() as extract_dir:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        func_codes.update(extract_functions_from_file(file_path))
        except Exception as e:
            logger.error(f"Error processing zip file {zip_path}: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing zip file: {str(e)}")
    
    return func_codes

def get_embedding_ollama(text: str) -> np.ndarray:
    """Generate embedding using Ollama's local model (Llama 3.2 3B)."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": "llama3.2:3b", "prompt": text},
            timeout=30
        )
        if response.status_code == 404:
            raise HTTPException(
                status_code=500,
                detail="Ollama API endpoint not found. Ensure the server is running and the URL is correct."
            )
        elif response.status_code == 400:
            raise HTTPException(
                status_code=500,
                detail="Bad request to Ollama API. Check if the model 'llama3.2:3b' is available."
            )
        response.raise_for_status()
        return np.array(response.json()["embedding"])
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=500,
            detail="Cannot connect to Ollama server. Ensure it is running on localhost:11434."
        )
    except Exception as e:
        logger.error(f"Ollama embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama embedding generation failed: {str(e)}")

def get_embedding_openai(text: str) -> np.ndarray:
    """Generate embedding using OpenAI's API."""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
        )
    
    try:
        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-ada-002",
                "input": text
            },
            timeout=10
        )
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"])
    except Exception as e:
        logger.error(f"OpenAI embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI embedding generation failed: {str(e)}")

def get_embedding(text: str, model: EmbeddingModel = EmbeddingModel.OLLAMA) -> np.ndarray:
    """Generate embedding based on selected model."""
    if model == EmbeddingModel.OLLAMA:
        return get_embedding_ollama(text)
    else:
        return get_embedding_openai(text)

def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    try:
        return float(1 - cosine(emb1, emb2))
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0

def store_embedding(func_name: str, embedding: np.ndarray, is_ideal: bool = True) -> None:
    """Store embedding in MongoDB."""
    try:
        collection.insert_one({
            "function_name": func_name,
            "embedding": embedding.tolist(),
            "is_ideal": is_ideal
        })
    except Exception as e:
        logger.error(f"Error storing embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to store embedding")

@app.post("/evaluate")
async def evaluate(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...),
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA)
) -> Dict[str, Any]:
    """Handle file uploads and perform evaluation."""
    if model == EmbeddingModel.OPENAI and not OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OpenAI model selected but API key not configured. Please set OPENAI_API_KEY in .env file."
        )

    try:
        # Create temporary files for uploads
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as sub_temp:
            await submission.seek(0)
            content = await submission.read()
            sub_temp.write(content)
            sub_path = sub_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as ideal_temp:
            await ideal.seek(0)
            content = await ideal.read()
            ideal_temp.write(content)
            ideal_path = ideal_temp.name

        try:
            # Extract functions from both zip files
            ideal_funcs = extract_functions_from_zip(ideal_path)
            submission_funcs = extract_functions_from_zip(sub_path)

            if not ideal_funcs:
                raise HTTPException(status_code=400, detail="No Python functions found in ideal answer")
            if not submission_funcs:
                raise HTTPException(status_code=400, detail="No Python functions found in submission")

            # Clear previous embeddings
            collection.delete_many({})

            # Generate and store embeddings for ideal answer
            ideal_embeddings = {}
            for name, code in ideal_funcs.items():
                embedding = get_embedding(code, model)
                store_embedding(name, embedding, is_ideal=True)
                ideal_embeddings[name] = embedding

            # Generate embeddings for submission
            submission_embeddings = {
                name: get_embedding(code, model)
                for name, code in submission_funcs.items()
            }

            # Compare functions and build report
            report = {}
            total_similarity = 0
            func_count = len(ideal_funcs)
            
            for func_name in ideal_funcs:
                if func_name in submission_funcs:
                    similarity = compute_similarity(
                        ideal_embeddings[func_name],
                        submission_embeddings[func_name]
                    )
                    status = "Correct" if similarity > SIMILARITY_THRESHOLD else "Incorrect"
                    report[func_name] = {
                        "status": status,
                        "similarity": round(similarity, 2)
                    }
                    total_similarity += similarity
                else:
                    report[func_name] = {"status": "Missing", "similarity": 0}

            # Calculate overall score
            overall_score = round((total_similarity / func_count * 100) if func_count > 0 else 0, 2)

            return {
                "report": report,
                "overall_score": f"{overall_score}%",
                "total_functions": func_count,
                "matched_functions": len([f for f in report if report[f]["status"] != "Missing"]),
                "model_used": model
            }

        finally:
            # Clean up temporary files
            for path in [sub_path, ideal_path]:
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Error removing temporary file {path}: {e}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def main_page():
    """Serve the upload interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>AI Assignment Checker</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }
                .upload-form {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .form-group {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                label {
                    font-weight: bold;
                    color: #555;
                }
                input[type="file"], select {
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                button {
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                    transition: background-color 0.3s;
                }
                button:hover {
                    background-color: #45a049;
                }
                #result {
                    margin-top: 20px;
                    padding: 20px;
                    border-radius: 4px;
                    display: none;
                }
                .loading {
                    text-align: center;
                    display: none;
                }
                .model-info {
                    margin-top: 10px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Assignment Checker</h1>
                <form id="uploadForm" class="upload-form">
                    <div class="form-group">
                        <label>Student Submission (ZIP):</label>
                        <input type="file" name="submission" accept=".zip" required>
                    </div>
                    <div class="form-group">
                        <label>Ideal Answer (ZIP):</label>
                        <input type="file" name="ideal" accept=".zip" required>
                    </div>
                    <div class="form-group">
                        <label>Select Embedding Model:</label>
                        <select name="model" required>
                            <option value="ollama">Ollama (Local)</option>
                            <option value="openai">OpenAI (API)</option>
                        </select>
                        <div class="model-info">
                            <strong>Model Information:</strong><br>
                            - Ollama: Uses local processing, no API key needed<br>
                            - OpenAI: Better accuracy, requires API key configuration
                        </div>
                    </div>
                    <button type="submit">Evaluate</button>
                </form>
                <div id="loading" class="loading">
                    Evaluating submission... Please wait...
                </div>
                <div id="result"></div>
            </div>

            <script>
                document.getElementById('uploadForm').onsubmit = async (e) => {
                    e.preventDefault();
                    
                    const formData = new FormData(e.target);
                    
                    const loading = document.getElementById('loading');
                    const result = document.getElementById('result');
                    
                    loading.style.display = 'block';
                    result.style.display = 'none';
                    
                    try {
                        const response = await fetch('/evaluate', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            let resultHtml = `<h3>Evaluation Results</h3>
                                            <p><strong>Model Used:</strong> ${data.model_used}</p>
                                            <p><strong>Overall Score:</strong> ${data.overall_score}</p>
                                            <p><strong>Functions Analyzed:</strong> ${data.total_functions}</p>
                                            <p><strong>Functions Matched:</strong> ${data.matched_functions}</p>
                                            <h4>Detailed Report:</h4><ul>`;
                            
                            for (const [func, details] of Object.entries(data.report)) {
                                const color = details.status === 'Correct' ? 'green' : 
                                            details.status === 'Missing' ? 'red' : 'orange';
                                resultHtml += `<li><strong>${func}:</strong> 
                                             <span style="color: ${color}">${details.status}</span>
                                             (Similarity: ${details.similarity * 100}%)</li>`;
                            }
                            resultHtml += '</ul>';
                            result.innerHTML = resultHtml;
                        } else {
                            result.innerHTML = `<p style="color: red">Error: ${data.detail}</p>`;
                        }
                    } catch (error) {
                        result.innerHTML = `<p style="color: red">Error: ${error.message}</p>`;
                    } finally {
                        loading.style.display = 'none';
                        result.style.display = 'block';
                    }
                };
            </script>
        </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port) 