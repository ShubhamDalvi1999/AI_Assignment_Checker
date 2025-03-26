import ast
import zipfile
import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
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
from rag_processor import RAGProcessor
from datetime import datetime, UTC
from utils.tokenizer_utils import count_tokens

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

# API URLs and keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/embeddings"
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

# Initialize RAG processor
rag_processor = RAGProcessor(mongo_client)

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

def _get_embedding_ollama(text: str) -> np.ndarray:
    """Generate embedding using Ollama's local model (Llama 3.2 3B).
    
    This function sends the provided text to a locally running Ollama server
    and retrieves a vector embedding representation of the text using the
    Llama 3.2 3B model. The embedding is returned as a numpy array.
    
    Request format:
    ```
    POST http://localhost:11434/api/embeddings
    Content-Type: application/json
    
    {
        "model": "llama3.2:3b",
        "prompt": "<text to embed>"
    }
    ```
    
    Response format:
    ```
    {
        "embedding": [0.123, 0.456, ...],  # Vector of floating point values
        "model": "llama3.2:3b"
    }
    ```
    
    Error handling:
    - ConnectionError: If Ollama server is not running
    - 404 Error: If API endpoint is incorrect
    - 400 Error: If model is not available
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        numpy.ndarray: Vector embedding of the input text
    """
    try:
        # Log the start of embedding generation
        logger.info(f"Generating Ollama embedding for text of length {len(text)}")
        
        # Check if text is empty or too short
        if not text or len(text.strip()) < 1:
            logger.warning("Empty or too short text provided for embedding")
            raise ValueError("Text must not be empty")
            
        # Check if Ollama server is running
        try:
            health_check = requests.get(OLLAMA_BASE_URL, timeout=5)
            if health_check.status_code != 200:
                logger.error("Ollama server is not responding properly")
                raise ConnectionError("Ollama server is not responding properly")
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama server")
            raise ConnectionError("Cannot connect to Ollama server. Ensure Ollama is running on localhost:11434")
        
        # Generate embedding
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": "llama3.2:3b", "prompt": text},
            timeout=30
        )
        
        # Handle specific error cases
        if response.status_code == 404:
            logger.error("Ollama API endpoint not found")
            raise HTTPException(
                status_code=500,
                detail="Ollama API endpoint not found. Ensure the server is running and the URL is correct."
            )
        elif response.status_code == 400:
            logger.error("Bad request to Ollama API")
            raise HTTPException(
                status_code=500,
                detail="Bad request to Ollama API. Check if the model 'llama3.2:3b' is available."
            )
        elif response.status_code != 200:
            logger.error(f"Unexpected status code from Ollama API: {response.status_code}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected response from Ollama API: {response.text}"
            )
            
        # Parse response
        result = response.json()
        if "embedding" not in result:
            logger.error("No embedding in Ollama response")
            raise ValueError("No embedding in Ollama response")
            
        embedding = np.array(result["embedding"])
        logger.info(f"Successfully generated embedding of shape {embedding.shape}")
        return embedding
        
    except requests.exceptions.Timeout:
        logger.error("Timeout while generating Ollama embedding")
        raise HTTPException(
            status_code=500,
            detail="Timeout while generating embedding. The model might be too slow or overloaded."
        )
    except requests.exceptions.ConnectionError:
        logger.error("Connection error while generating Ollama embedding")
        raise HTTPException(
            status_code=500,
            detail="Cannot connect to Ollama server. Ensure it is running on localhost:11434."
        )
    except Exception as e:
        logger.error(f"Ollama embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama embedding generation failed: {str(e)}")

def _get_embedding_openai(text: str) -> np.ndarray:
    """Generate embedding using OpenAI's API.
    
    This function sends the provided text to OpenAI's embedding API
    and retrieves a vector embedding using the text-embedding-ada-002 model.
    The API requires authentication with an API key and returns a 1536-dimensional
    embedding vector.
    
    Request format:
    ```
    POST https://api.openai.com/v1/embeddings
    Headers:
        Authorization: Bearer <OPENAI_API_KEY>
        Content-Type: application/json
    
    {
        "model": "text-embedding-ada-002",
        "input": "<text to embed>"
    }
    ```
    
    Response format:
    ```
    {
        "data": [
            {
                "embedding": [0.123, 0.456, ...],  # 1536-dimensional vector
                "index": 0,
                "object": "embedding"
            }
        ],
        "model": "text-embedding-ada-002",
        "object": "list",
        "usage": {
            "prompt_tokens": <number of tokens in input>,
            "total_tokens": <number of tokens in input>
        }
    }
    ```
    
    Error handling:
    - Missing API key
    - Network errors
    - API rate limits
    - Invalid requests
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        numpy.ndarray: 1536-dimensional vector embedding of the input text
    """
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
        return _get_embedding_ollama(text)
    else:
        return _get_embedding_openai(text)

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
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA),
    use_openai_feedback: bool = Form(False)
) -> Dict[str, Any]:
    """Handle file uploads and perform evaluation."""
    
    # Check for OpenAI API key if OpenAI model is selected
    if model == EmbeddingModel.OPENAI and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable."
        )
        
    # Create temporary files to extract the uploads
    temp_dir_student = tempfile.mkdtemp()
    temp_dir_ideal = tempfile.mkdtemp()

    try:
        # Update RAG processor configuration for feedback generation
        rag_processor.use_openai = use_openai_feedback
        
        # TEMPORARY FILE HANDLING:
        # File Path Requirement: The zipfile library needs a filesystem path to extract ZIP contents, but FastAPI's UploadFile only provides a file-like object in memory, not a path.
        # Persistence During Processing: The uploaded files exist temporarily in memory, but we need them to persist while we extract the ZIP contents, process all Python files, parse and extract functions, and generate embeddings.
        # Windows Compatibility: Windows has special file-locking behavior where open files can't be modified by other processes. Using delete=False and manual cleanup ensures compatibility.
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as student_temp_file:
            await submission.seek(0)
            content = await submission.read()
            student_temp_file.write(content)
            student_zip_path = student_temp_file.name  # Get path for later extraction

        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as ideal_temp:
            await ideal.seek(0)
            content = await ideal.read()
            ideal_temp.write(content)
            ideal_path = ideal_temp.name  # Get path for later extraction

        try:
            # FUNCTION EXTRACTION PROCESS:
            # 1. Unzip archives to temporary directories
            # 2. Find all .py files in the extracted directories
            # 3. Parse each file to extract function definitions using AST
            # 4. Return dictionary of {function_name: function_code}
            ideal_funcs = extract_functions_from_zip(ideal_path)    # Instructor/reference solution
            student_functions = extract_functions_from_zip(student_zip_path) # Student submission

            if not ideal_funcs:
                raise HTTPException(status_code=400, detail="No Python functions found in ideal answer")
            if not student_functions:
                raise HTTPException(status_code=400, detail="No Python functions found in submission")

            # IMPORTANT: Delete all previous embeddings from MongoDB.
            # This creates a clean slate for the current evaluation and means:
            # 1. Each evaluation session works in isolation
            # 2. No historical data is maintained between sessions
            # 3. The retrieval corpus will only contain embeddings from this evaluation
            # This is a limitation: student embeddings from previous sessions cannot be leveraged
            collection.delete_many({})

            # IDEAL CODE EMBEDDING PROCESS:
            # Generate and store embeddings for ideal/reference solutions
            # These embeddings will be used as the retrieval corpus for RAG
            ideal_embeddings = {}
            for name, code in ideal_funcs.items():
                # Generate embedding using selected model (Ollama or OpenAI)
                embedding = get_embedding(code, model)
                # Store in MongoDB with RAG processor to allow retrieval during comparison
                # Note: Only ideal code embeddings are stored in the database
                rag_processor.store_code_context(
                    name,
                    code,
                    embedding,
                    {"is_ideal": True, "timestamp": datetime.now(UTC)}
                )
                ideal_embeddings[name] = embedding

            # STUDENT CODE EMBEDDING PROCESS:
            # Generate all student embeddings at once before entering the comparison loop
            # This reduces API calls and improves overall processing speed
            student_embeddings = {}
            for func_name, code in student_functions.items():
                # Only generate embeddings for functions that exist in ideal solution
                if func_name in ideal_funcs:
                    student_embeddings[func_name] = get_embedding(code, model)
            

            # STUDENT CODE COMPARISON PROCESS:
            # Process each function and generate the comparison report
            function_reports = {}
            total_similarity = 0
            func_count = len(ideal_funcs)
            
            for func_name, ideal_code in ideal_funcs.items():
                if func_name in student_functions:
                    # Function exists in both - compare them
                    student_code = student_functions[func_name]
                    
                    # Get embeddings we generated earlier
                    student_embedding = student_embeddings[func_name]
                    ideal_embedding = ideal_embeddings[func_name]
                    
                    # Calculate similarity
                    similarity = 1 - cosine(student_embedding, ideal_embedding)
                    logger.info(f"Function {func_name}: similarity = {similarity:.4f}")
                    
                    # Generate detailed report with RAG processor
                    detailed_report = rag_processor.generate_comparison_report(
                        student_code, 
                        ideal_code, 
                        student_embedding, 
                        ideal_embedding
                    )
                    
                    # Create report with RAG-generated feedback
                    function_reports[func_name] = {
                        "status": "Correct" if similarity >= SIMILARITY_THRESHOLD else "Incorrect",
                        "similarity": float(similarity),
                        "structure_analysis": detailed_report["structure_analysis"],
                        "recommendations": detailed_report["recommendations"],
                        "similar_contexts": [
                            {
                                "function_name": ctx.get("function_name", "Unknown"),
                                "similarity": float(ctx.get("similarity", 0))
                            } for ctx in detailed_report["similar_contexts"]
                        ],
                        "feedback": detailed_report.get("feedback", "No feedback generated.")
                    }
                    
                    total_similarity += similarity
                else:
                    # Function in ideal but not in student submission
                    function_reports[func_name] = {
                        "status": "Missing",
                        "similarity": 0.0,
                        "structure_analysis": {
                            "variables": {"missing_variables": [], "extra_variables": []},
                            "control_flow": {"missing_control_structures": [], "extra_control_structures": []},
                            "function_calls": {"missing_calls": [], "extra_calls": []}
                        },
                        "recommendations": ["Implement this required function"],
                        "similar_contexts": [],
                        "feedback": "This function is missing from your submission."
                    }

            # Calculate overall score
            overall_score = round((total_similarity / func_count * 100) if func_count > 0 else 0, 2)

            # At this point, the student embeddings no longer exist as they were local variables
            # Only the ideal embeddings remain in MongoDB until the next evaluation, 
            # when they will be cleared by collection.delete_many({})

            return {
                "report": function_reports,
                "overall_score": f"{overall_score}%",
                "total_functions": func_count,
                "matched_functions": len([f for f in function_reports if function_reports[f]["status"] != "Missing"]),
                "model_used": model
            }

        finally:
            # CLEANUP PROCESS:
            # Remove temporary ZIP files to prevent storage leaks
            # Both student and ideal submission files are deleted here
            for path in [student_zip_path, ideal_path]:
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Error removing temporary file {path}: {e}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/estimate-tokens")
async def estimate_tokens(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...)
) -> Dict[str, Any]:
    """Estimate token usage for the uploaded files."""
    try:
        # TEMPORARY FILE HANDLING:
        # File Path Requirement: The zipfile library needs a filesystem path to extract ZIP contents, but FastAPI's UploadFile only provides a file-like object in memory, not a path.
        # Persistence During Processing: The uploaded files exist temporarily in memory, but we need them to persist while we extract the ZIP contents, process all Python files, parse and extract functions, and generate embeddings.
        # Windows Compatibility: Windows has special file-locking behavior where open files can't be modified by other processes. Using delete=False and manual cleanup ensures compatibility.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as student_temp_file:
            await submission.seek(0)
            content = await submission.read()
            student_temp_file.write(content)
            student_zip_path = student_temp_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as ideal_temp:
            await ideal.seek(0)
            content = await ideal.read()
            ideal_temp.write(content)
            ideal_path = ideal_temp.name

        try:
            # Extract functions from both ZIP files
            student_functions = extract_functions_from_zip(student_zip_path)
            ideal_funcs = extract_functions_from_zip(ideal_path)

            # Calculate token counts
            student_token_total = sum([count_tokens(code) for code in student_functions.values()])
            ideal_token_total = sum([count_tokens(code) for code in ideal_funcs.values()])
            
            # Estimate overhead for structure and retrieval (approximate)
            structure_overhead = 5000  # Approximate tokens for structure analysis JSON
            retrieval_overhead = 10000  # Approximate tokens for retrieved similar contexts
            prompt_overhead = 2000     # Approximate tokens for prompt template
            
            total_estimate = student_token_total + ideal_token_total + structure_overhead + retrieval_overhead + prompt_overhead

            # Determine safety status
            is_safe = total_estimate <= 100000
            
            return {
                "student_tokens": student_token_total,
                "ideal_tokens": ideal_token_total,
                "structure_overhead": structure_overhead,
                "retrieval_overhead": retrieval_overhead,
                "prompt_overhead": prompt_overhead,
                "total_estimate": total_estimate,
                "is_safe": is_safe,
                "student_functions": len(student_functions),
                "ideal_functions": len(ideal_funcs)
            }

        finally:
            # Clean up temporary files
            for path in [student_zip_path, ideal_path]:
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Error removing temporary file {path}: {e}")

    except Exception as e:
        logger.error(f"Token estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Token estimation failed: {str(e)}")

@app.get("/")
async def main_page(request: Request, submission: str = None, ideal: str = None, model: str = None):
    html_content = r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Assignment Checker</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="file"] {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            select {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            #loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            #result {
                margin-top: 20px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: none;
            }
            .error-message {
                color: #dc3545;
                padding: 10px;
                border: 1px solid #dc3545;
                border-radius: 4px;
                margin: 10px 0;
            }
            .openai-warning {
                color: #856404;
                background-color: #fff3cd;
                border: 1px solid #ffeeba;
                border-radius: 4px;
                padding: 10px;
                margin: 10px 0;
                font-size: 0.9em;
            }
            .progress-container {
                margin: 20px 0;
                display: none;
            }
            .progress-step {
                padding: 10px;
                margin: 5px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: flex;
                align-items: center;
            }
            .progress-step.active {
                background-color: #e3f2fd;
                border-color: #2196f3;
            }
            .progress-step.completed {
                background-color: #e8f5e9;
                border-color: #4caf50;
            }
            .progress-step.error {
                background-color: #ffebee;
                border-color: #f44336;
            }
            .step-icon {
                margin-right: 10px;
            }
            .step-label {
                font-weight: bold;
                flex-grow: 1;
            }
            .step-status {
                color: #666;
            }
            .function-report {
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .status-correct {
                color: #4caf50;
            }
            .status-incorrect {
                color: #f44336;
            }
            .status-missing {
                color: #ff9800;
            }
            .similar-contexts {
                margin: 20px 0;
            }
            .feedback-section {
                margin: 20px 0;
                padding: 15px;
                background-color: #f1f8e9;
                border-left: 4px solid #8bc34a;
                border-radius: 4px;
            }
            .feedback-content {
                margin: 0;
                padding: 10px;
                background-color: #ffffff;
                border-radius: 4px;
                white-space: pre-wrap;
                font-family: monospace;
            }
            pre {
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Assignment Checker</h1>
            <form id="uploadForm">
                <div class="form-group">
                    <label for="submission">Student Submission (ZIP):</label>
                    <input type="file" id="submission" name="submission" accept=".zip" required>
                </div>
                <div class="form-group">
                    <label for="ideal">Ideal Solution (ZIP):</label>
                    <input type="file" id="ideal" name="ideal" accept=".zip" required>
                </div>
                <div class="form-group">
                    <label for="model">Model:</label>
                    <select id="model" name="model" required>
                        <option value="ollama">Ollama (Local)</option>
                        <option value="openai">OpenAI</option>
                    </select>
                </div>
                <button type="submit" id="evaluateBtn">Evaluate</button>
            </form>
            <div id="loading">
                <div class="spinner"></div>
                <p>Processing your submission...</p>
            </div>
            <div id="progressContainer" class="progress-container">
                <h3>Progress</h3>
                <div id="progressSteps"></div>
            </div>
            <div id="result"></div>
        </div>
        <script>
            // Progress tracking
            const progressSteps = [
                { id: 'fileProcessing', label: 'Processing ZIP files', details: 'Extracting and analyzing Python files' },
                { id: 'embeddingGeneration', label: 'Generating embeddings', details: 'Creating vector representations of code' },
                { id: 'comparison', label: 'Comparing code', details: 'Analyzing code similarity and structure' },
                { id: 'feedback', label: 'Generating feedback', details: 'Creating detailed feedback and recommendations' },
                { id: 'finalization', label: 'Finalizing results', details: 'Preparing final evaluation report' }
            ];

            // Initialize progress container
            const progressContainer = document.getElementById('progressContainer');
            const progressStepsContainer = document.getElementById('progressSteps');
            progressSteps.forEach(step => {
                const stepElement = document.createElement('div');
                stepElement.id = step.id;
                stepElement.className = 'progress-step';
                stepElement.innerHTML = `
                    <span class="step-icon">⏳</span>
                    <span class="step-label">${step.label}</span>
                    <span class="step-details">${step.details}</span>
                    <span class="step-status">Pending</span>
                `;
                progressStepsContainer.appendChild(stepElement);
            });

            // Form submission handling
            const uploadForm = document.getElementById('uploadForm');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const evaluateBtn = document.getElementById('evaluateBtn');
            const modelSelect = document.getElementById('model');

            // OpenAI model selection warning
            modelSelect.addEventListener('change', function() {
                if (this.value === 'openai') {
                    // Show warning about OpenAI API key
                    if (!document.getElementById('openai-warning')) {
                        const warning = document.createElement('div');
                        warning.id = 'openai-warning';
                        warning.className = 'openai-warning';
                        warning.innerHTML = `
                            <p><strong>⚠️ Warning:</strong> OpenAI model requires an API key to be configured in the server's environment variables.</p>
                            <p>If no API key is configured, the evaluation will fail.</p>
                        `;
                        this.parentNode.appendChild(warning);
                    }
                } else {
                    // Remove warning if switching back to Ollama
                    const warning = document.getElementById('openai-warning');
                    if (warning) {
                        warning.remove();
                    }
                }
            });

            uploadForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                // Show loading state and progress container
                loading.style.display = 'block';
                progressContainer.style.display = 'block';
                result.style.display = 'none';
                evaluateBtn.disabled = true;

                // Reset progress steps
                progressSteps.forEach(step => {
                    const stepElement = document.getElementById(step.id);
                    stepElement.className = 'progress-step';
                    stepElement.querySelector('.step-icon').textContent = '⏳';
                    stepElement.querySelector('.step-status').textContent = 'Pending';
                });

                // Update first step to active
                const firstStep = document.getElementById('fileProcessing');
                firstStep.className = 'progress-step active';
                firstStep.querySelector('.step-icon').textContent = '🔄';
                firstStep.querySelector('.step-status').textContent = 'Processing';

                const formData = new FormData(uploadForm);
                
                try {
                    console.log('Submitting form...');
                    const response = await fetch('/evaluate', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error(`Error: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    console.log('Received response:', data);

                    // Update progress steps to completed
                    progressSteps.forEach(step => {
                        const stepElement = document.getElementById(step.id);
                        stepElement.className = 'progress-step completed';
                        stepElement.querySelector('.step-icon').textContent = '✅';
                        stepElement.querySelector('.step-status').textContent = 'Completed';
                    });

                    // Format and display results
                    let resultHtml = `
                        <h2>Evaluation Results</h2>
                        <p><strong>Overall Score:</strong> ${data.overall_score}</p>
                        <p><strong>Total Functions:</strong> ${data.total_functions}</p>
                        <p><strong>Matched Functions:</strong> ${data.matched_functions}</p>
                        <p><strong>Model Used:</strong> ${data.model_used}</p>
                    `;

                    // Display function reports
                    for (const [funcName, report] of Object.entries(data.report)) {
                        resultHtml += `
                            <div class="function-report">
                                <h3>Function: ${funcName}</h3>
                                <p class="status-${report.status.toLowerCase()}">
                                    Status: ${report.status} (Similarity: ${report.similarity})
                                </p>
                                
                                <div class="structure-analysis">
                                    <h4>Structure Analysis:</h4>
                                    <pre>${JSON.stringify(report.structure_analysis, null, 2)}</pre>
                                </div>
                                
                                <div class="recommendations">
                                    <h4>Recommendations:</h4>
                                    <ul>
                                        ${report.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                                    </ul>
                                </div>
                                
                                <div class="similar-contexts">
                                    <h4>Similar Implementations:</h4>
                                    <pre>${JSON.stringify(report.similar_contexts, null, 2)}</pre>
                                </div>

                                ${report.feedback ? `
                                <div class="feedback-section">
                                    <h4>AI Feedback:</h4>
                                    <pre class="feedback-content">${report.feedback}</pre>
                                </div>
                                ` : ''}
                            </div>
                        `;
                    }

                    result.innerHTML = resultHtml;
                    result.style.display = 'block';
                } catch (error) {
                    console.error('Evaluation error:', error);
                    
                    // Update progress steps to error state
                    progressSteps.forEach(step => {
                        const stepElement = document.getElementById(step.id);
                        stepElement.className = 'progress-step error';
                        stepElement.querySelector('.step-icon').textContent = '❌';
                        stepElement.querySelector('.step-status').textContent = 'Error';
                    });

                    result.innerHTML = `
                        <div class="error-message">
                            <h3>Error During Evaluation</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                    result.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                    evaluateBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port) 