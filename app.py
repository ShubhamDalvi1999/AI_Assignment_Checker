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
from typing import Dict, Any, Optional, Tuple
import tempfile
from enum import Enum
from dotenv import load_dotenv
import time
import random

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available, using approximate token counting")

# Load environment variables
load_dotenv()

# Configure logging with file handlers
import logging.handlers
import os.path

# Set up logs directory
try:
    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")
    
    # Create logs directory with full path
    logs_dir = os.path.join(current_dir, "logs")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
        print(f"Created logs directory at: {logs_dir}")
    else:
        print(f"Logs directory already exists at: {logs_dir}")
    
    # Verify the directory exists and is writable
    if not os.access(logs_dir, os.W_OK):
        print(f"WARNING: Logs directory {logs_dir} is not writable")
        # Fall back to a directory we know should be writable
        logs_dir = os.path.join(os.path.expanduser("~"), "ai_assignment_checker_logs")
        print(f"Falling back to: {logs_dir}")
        os.makedirs(logs_dir, exist_ok=True)
except Exception as e:
    print(f"Error creating logs directory: {e}")
    # Fall back to home directory or temp directory as a last resort
    import tempfile
    logs_dir = os.path.join(tempfile.gettempdir(), "ai_assignment_checker_logs")
    print(f"Using temporary logs directory: {logs_dir}")
    os.makedirs(logs_dir, exist_ok=True)

# Configure the root logger
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level))

# Clear any existing handlers (to avoid duplicate logs)
if root_logger.handlers:
    root_logger.handlers.clear()

# Create formatters
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(thread)d - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, log_level))
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

try:
    # General log file with rotation (10 MB per file, keep 30 days of logs)
    general_log_file = os.path.join(logs_dir, "app.log")
    print(f"Creating general log file at: {general_log_file}")
    general_file_handler = logging.handlers.TimedRotatingFileHandler(
        general_log_file, 
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    general_file_handler.setLevel(getattr(logging, log_level))
    general_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(general_file_handler)
    print(f"Successfully added general log handler")

    # Error log file with rotation (for ERROR and above)
    error_log_file = os.path.join(logs_dir, "error.log")
    error_file_handler = logging.handlers.RotatingFileHandler(
        error_log_file, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=10,
        encoding='utf-8'
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_file_handler)
    print(f"Successfully added error log handler")

    # API call log file with rotation
    api_log_file = os.path.join(logs_dir, "api_calls.log")
    api_file_handler = logging.handlers.RotatingFileHandler(
        api_log_file, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=10,
        encoding='utf-8'
    )
    api_file_handler.setLevel(logging.INFO)
    # Only capture API-related logs
    class APICallFilter(logging.Filter):
        def filter(self, record):
            return 'API' in record.getMessage() or 'OpenAI' in record.getMessage() or 'Ollama' in record.getMessage()
    api_file_handler.addFilter(APICallFilter())
    api_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(api_file_handler)
    print(f"Successfully added API log handler")
    
    print(f"Logging fully configured with output to {logs_dir}")
except Exception as e:
    print(f"Error setting up log files: {e}")
    # Ensure at least console logging works
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.handlers = [console_handler]
    print("Falling back to console-only logging")

# Get the logger for this file
logger = logging.getLogger(__name__)
logger.info("Logging system initialized")

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

def get_embedding_openai(text: str, max_retries: int = 1) -> Tuple[np.ndarray, int]:
    """
    Generate embedding using OpenAI's API with retry logic for rate limits.
    
    Args:
        text: The text to generate embeddings for
        max_retries: Maximum number of retries for rate limits
        
    Returns:
        Tuple of (embedding array, token count)
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_api_key_here":
        logger.error("OpenAI API key is missing or using the default placeholder value")
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please update the OPENAI_API_KEY in your .env file with a valid key from https://platform.openai.com/api-keys"
        )
    
    # Count tokens for logging and tracking purposes
    token_count = count_tokens(text)
    logger.info(f"Sending {token_count} tokens to OpenAI API")
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Debug information
            logger.info(f"Making OpenAI API request with API key: {OPENAI_API_KEY[:5]}{'*' * 10}")
            
            response = requests.post(
                OPENAI_API_URL,
                headers=headers,
                json={
                    "model": "text-embedding-ada-002",
                    "input": text
                },
                timeout=10
            )
            
            # Log the response status and headers for debugging
            logger.info(f"OpenAI API response status: {response.status_code}")
            logger.info(f"OpenAI API response headers: {dict(response.headers)}")
            
            if response.status_code == 401:
                logger.error(f"Authentication error with OpenAI API: {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid OpenAI API key or authentication error. Error details: {response.text}"
                )
            elif response.status_code == 429:
                retry_count += 1
                rate_limit_info = response.headers.get('x-ratelimit-remaining', 'unknown')
                reset_time = response.headers.get('x-ratelimit-reset-tokens', 'unknown')
                logger.warning(f"Rate limit hit. Remaining: {rate_limit_info}, Reset: {reset_time}")
                
                if retry_count >= max_retries:
                    logger.error(f"OpenAI API rate limit exceeded after {max_retries} retries: {str(e)}")
                    raise HTTPException(
                        status_code=429, 
                        detail=f"OpenAI API rate limit exceeded after {max_retries} attempts: {str(e)}. Please wait before making more requests or upgrade your OpenAI account limits."
                    )
                
                # Exponential backoff with jitter
                wait_time = (2 ** retry_count) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit, retrying once in {wait_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif response.status_code != 200:
                # Log other non-200 responses
                logger.error(f"OpenAI API error: {response.status_code}, {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenAI API error: {response.status_code}, {response.text}"
                )
            
            try:
                response_data = response.json()
                if "data" not in response_data or not response_data["data"]:
                    logger.error(f"Unexpected response format from OpenAI: {response_data}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Unexpected response format from OpenAI: {response_data}"
                    )
                return np.array(response_data["data"][0]["embedding"]), token_count
            except (KeyError, IndexError, ValueError) as e:
                logger.error(f"Error parsing OpenAI response: {e}. Response: {response.text}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error parsing OpenAI response: {e}. Response: {response.text}"
                )
            
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"OpenAI API rate limit exceeded after {max_retries} retries: {str(e)}")
                    raise HTTPException(
                        status_code=429, 
                        detail=f"OpenAI API rate limit exceeded after {max_retries} attempts: {str(e)}. Please wait before making more requests or upgrade your OpenAI account limits."
                    )
                
                # Exponential backoff with jitter
                wait_time = (2 ** retry_count) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit, retrying once in {wait_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"OpenAI API request failed: {e}")
                raise HTTPException(status_code=500, detail=f"OpenAI embedding generation failed: {str(e)}")
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI embedding generation failed: {str(e)}")
    
    # This should generally not be reached due to the retry logic above
    raise HTTPException(status_code=500, detail="Failed to generate embeddings after maximum retries")

def get_embedding(text: str, model: EmbeddingModel = EmbeddingModel.OLLAMA) -> Tuple[np.ndarray, int]:
    """
    Generate embedding based on selected model.
    
    Args:
        text: The text to generate embeddings for
        model: The embedding model to use
        
    Returns:
        For OpenAI: Tuple of (embedding array, token count)
        For Ollama: Tuple of (embedding array, 0) - no token tracking for Ollama
    """
    if model == EmbeddingModel.OLLAMA:
        return get_embedding_ollama(text), 0  # No token counting for Ollama
    else:
        return get_embedding_openai(text)  # Returns (embedding, token_count)

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

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a string using tiktoken if available,
    or approximate if not.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        The number of tokens
    """
    if TIKTOKEN_AVAILABLE:
        # text-embedding-ada-002 uses cl100k_base encoding
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    else:
        # Simple approximation (4 chars ~= 1 token)
        return len(text) // 4

@app.post("/evaluate")
async def evaluate(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...),
    model: EmbeddingModel = Form(EmbeddingModel.OLLAMA)
) -> Dict[str, Any]:
    """Handle file uploads and perform evaluation."""
    if model == EmbeddingModel.OPENAI:
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_api_key_here":
            raise HTTPException(
                status_code=400,
                detail="OpenAI model selected but API key not configured or using default value. Please update the OPENAI_API_KEY in your .env file with a valid key from https://platform.openai.com/api-keys"
            )
    elif model == EmbeddingModel.OLLAMA:
        # Check if Ollama server is accessible
        try:
            response = requests.get(OLLAMA_API_URL.replace("/api/embeddings", "/api/version"), timeout=2)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail="Ollama server is not responding correctly. Make sure it's running and accessible."
                )
        except requests.exceptions.RequestException:
            raise HTTPException(
                status_code=500,
                detail="Cannot connect to Ollama server. Ensure it is running on localhost:11434."
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

            # Generate and store embeddings - with progress updates
            logger.info(f"Generating embeddings using {model} model")
            total_functions = len(ideal_funcs) + len(submission_funcs)
            progress_counter = 0
            total_tokens = 0  # Track total tokens used
            
            # Generate embeddings for ideal answer
            ideal_embeddings = {}
            
            # Different approach based on model
            if model == EmbeddingModel.OPENAI:
                # Use batching for OpenAI to reduce API calls
                batch_size = min(8, len(ideal_funcs))  # Adjust batch size as needed
                logger.info(f"Using batched processing with batch size {batch_size} for OpenAI")
                
                # Process in smaller batches to avoid rate limits
                for i in range(0, len(ideal_funcs.items()), batch_size):
                    batch_items = list(ideal_funcs.items())[i:i+batch_size]
                    for name, code in batch_items:
                        try:
                            progress_counter += 1
                            logger.info(f"Generating embedding for ideal function {name} ({progress_counter}/{total_functions})")
                            embedding, token_count = get_embedding(code, model)
                            total_tokens += token_count
                            store_embedding(name, embedding, is_ideal=True)
                            ideal_embeddings[name] = embedding
                            # Add a small delay between API calls
                            if model == EmbeddingModel.OPENAI and i + batch_size < len(ideal_funcs.items()):
                                time.sleep(0.5)  # 500ms delay between batches
                        except Exception as e:
                            logger.error(f"Error processing ideal function {name}: {e}")
                            raise
            else:
                # Regular processing for Ollama
                for name, code in ideal_funcs.items():
                    progress_counter += 1
                    logger.info(f"Generating embedding for ideal function {name} ({progress_counter}/{total_functions})")
                    embedding, _ = get_embedding(code, model)
                    store_embedding(name, embedding, is_ideal=True)
                    ideal_embeddings[name] = embedding

            # Generate embeddings for submission with similar approach
            submission_embeddings = {}
            if model == EmbeddingModel.OPENAI:
                # Use batching for OpenAI to reduce API calls
                batch_size = min(8, len(submission_funcs))
                
                # Process in smaller batches
                for i in range(0, len(submission_funcs.items()), batch_size):
                    batch_items = list(submission_funcs.items())[i:i+batch_size]
                    for name, code in batch_items:
                        try:
                            progress_counter += 1
                            logger.info(f"Generating embedding for submission function {name} ({progress_counter}/{total_functions})")
                            embedding, token_count = get_embedding(code, model)
                            total_tokens += token_count
                            submission_embeddings[name] = embedding
                            # Add a small delay between API calls
                            if model == EmbeddingModel.OPENAI and i + batch_size < len(submission_funcs.items()):
                                time.sleep(0.5)  # 500ms delay between batches
                        except Exception as e:
                            logger.error(f"Error processing submission function {name}: {e}")
                            raise
            else:
                # Regular processing for Ollama
                for name, code in submission_funcs.items():
                    progress_counter += 1
                    logger.info(f"Generating embedding for submission function {name} ({progress_counter}/{total_functions})")
                    embedding, _ = get_embedding(code, model)
                    submission_embeddings[name] = embedding

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

            # Prepare response
            response_data = {
                "report": report,
                "overall_score": f"{overall_score}%",
                "total_functions": func_count,
                "matched_functions": len([f for f in report if report[f]["status"] != "Missing"]),
                "model_used": model
            }
            
            # Add token usage information for OpenAI
            if model == EmbeddingModel.OPENAI:
                estimated_cost = (total_tokens / 1000) * 0.0001  # $0.0001 per 1K tokens for embedding model
                response_data.update({
                    "token_usage": {
                        "total_tokens": total_tokens,
                        "estimated_cost_usd": f"${estimated_cost:.6f}"
                    }
                })
                logger.info(f"Total tokens used: {total_tokens}, estimated cost: ${estimated_cost:.6f}")
                logger.info(f"Response data: {response_data}")
            
            return response_data

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

@app.post("/calculate-tokens")
async def calculate_tokens(
    submission: UploadFile = File(...),
    ideal: UploadFile = File(...)
) -> Dict[str, Any]:
    """Calculate tokens for the uploaded files without running full analysis."""
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

            # Calculate total tokens for all functions
            total_tokens = 0
            for name, code in ideal_funcs.items():
                total_tokens += count_tokens(code)
            
            for name, code in submission_funcs.items():
                total_tokens += count_tokens(code)
            
            # Calculate API usage metrics
            tokens_per_request = 8192  # text-embedding-ada-002 limit
            total_requests = -(-total_tokens // tokens_per_request)  # Ceiling division
            
            # OpenAI rate limits
            tokens_per_minute_limit = 40000  # 40k TPM
            requests_per_minute_limit = 100   # 100 RPM
            requests_per_day_limit = 2000     # 2k RPD
            
            # Calculate required time and potential rate limit issues
            minutes_required = max(
                total_tokens / tokens_per_minute_limit,
                total_requests / requests_per_minute_limit
            )
            minutes_required = max(1, round(minutes_required, 2))
            
            # Check if we'd hit the daily request limit
            days_required = total_requests / requests_per_day_limit
            
            # Determine if rate limits would be hit
            will_hit_rpm_limit = total_requests > requests_per_minute_limit
            will_hit_tpm_limit = total_tokens > tokens_per_minute_limit
            will_hit_daily_limit = days_required > 1
            
            return {
                "total_functions": len(ideal_funcs) + len(submission_funcs),
                "ideal_functions": len(ideal_funcs),
                "submission_functions": len(submission_funcs),
                "total_tokens": total_tokens,
                "total_requests": total_requests,
                "tokens_per_minute_limit": tokens_per_minute_limit,
                "requests_per_minute_limit": requests_per_minute_limit,
                "requests_per_day_limit": requests_per_day_limit,
                "minutes_required": minutes_required,
                "days_required": round(days_required, 2),
                "will_hit_rpm_limit": will_hit_rpm_limit, 
                "will_hit_tpm_limit": will_hit_tpm_limit,
                "will_hit_daily_limit": will_hit_daily_limit,
                "estimated_cost_usd": f"${(total_tokens / 1000 * 0.0001):.6f}",
                "tokenizer_used": "tiktoken" if TIKTOKEN_AVAILABLE else "approximation"
            }
        finally:
            # Clean up temporary files
            for path in [sub_path, ideal_path]:
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Error removing temporary file {path}: {e}")

    except Exception as e:
        logger.error(f"Token calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Token calculation failed: {str(e)}")

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
                button:disabled {
                    background-color: #cccccc;
                    cursor: not-allowed;
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
                .error-box {
                    background-color: #ffebee;
                    border: 1px solid #ffcdd2;
                    color: #b71c1c;
                    padding: 15px;
                    margin-top: 20px;
                    border-radius: 4px;
                    display: none;
                }
                .instructions {
                    background-color: #e8f5e9;
                    border: 1px solid #c8e6c9;
                    padding: 15px;
                    margin-top: 20px;
                    border-radius: 4px;
                }
                .token-usage {
                    margin-top: 15px;
                    padding: 10px;
                    background-color: #f0f4f8;
                    border-radius: 4px;
                    border: 1px solid #d0e0f0;
                }
                .calculation-result {
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #f0f8ff;
                    border: 1px solid #b3d9ff;
                    border-radius: 4px;
                    display: none;
                }
                .warning {
                    color: #d32f2f;
                    font-weight: bold;
                }
                .safe {
                    color: #388e3c;
                    font-weight: bold;
                }
                .buttons-container {
                    display: flex;
                    gap: 10px;
                }
                #calculateButton {
                    background-color: #3f51b5;
                    display: none;
                }
                #calculateButton:hover {
                    background-color: #303f9f;
                }
                .infobox {
                    background-color: #e3f2fd;
                    border: 1px solid #90caf9;
                    padding: 10px 15px;
                    border-radius: 4px;
                    margin-top: 12px;
                    font-size: 0.9em;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .secondary-button {
                    background-color: #607d8b;
                    color: white;
                    padding: 8px 16px;
                    margin-top: 10px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                    transition: background-color 0.3s;
                }
                .secondary-button:hover {
                    background-color: #455a64;
                }
                .api-test-success {
                    background-color: #e8f5e9;
                    border: 1px solid #c8e6c9;
                    padding: 10px;
                    border-radius: 4px;
                    color: #388e3c;
                }
                .api-test-error {
                    background-color: #ffebee;
                    border: 1px solid #ffcdd2;
                    padding: 10px;
                    border-radius: 4px;
                    color: #d32f2f;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Assignment Checker</h1>
                <form id="uploadForm" class="upload-form">
                    <div class="form-group">
                        <label>Student Submission (ZIP):</label>
                        <input type="file" name="submission" id="submissionFile" accept=".zip" required>
                    </div>
                    <div class="form-group">
                        <label>Ideal Answer (ZIP):</label>
                        <input type="file" name="ideal" id="idealFile" accept=".zip" required>
                    </div>
                    <div class="form-group">
                        <label>Select Embedding Model:</label>
                        <select name="model" id="modelSelect" required>
                            <option value="ollama">Ollama (Local)</option>
                            <option value="openai">OpenAI (API)</option>
                        </select>
                        <div class="model-info">
                            <strong>Model Information:</strong><br>
                            - Ollama: Uses local processing, no API key needed<br>
                            - OpenAI: Better accuracy, requires API key configuration
                        </div>
                    </div>
                    <div id="apiKeyInfo" class="instructions" style="display: none;">
                        <strong>OpenAI API Key Setup:</strong><br>
                        1. Get your API key from <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI Platform</a><br>
                        2. Update the OPENAI_API_KEY in your .env file<br>
                        3. Restart the application<br><br>
                        <strong>Important Note About Rate Limits:</strong><br>
                        - Free API keys have lower rate limits (limited tokens per minute)<br>
                        - Processing many functions may exceed these limits<br>
                        - The app will automatically retry once when rate limited<br>
                        - For large assignments, consider a paid OpenAI account
                        <button type="button" id="testApiKeyButton" class="secondary-button">Test OpenAI API Key</button>
                        <div id="apiKeyTestResult" style="margin-top: 10px; display: none;"></div>
                    </div>
                    <div class="buttons-container">
                        <button type="submit">Evaluate</button>
                        <button type="button" id="calculateButton">Calculate OpenAI Usage</button>
                    </div>
                </form>
                <div id="loading" class="loading">
                    Processing... Please wait...
                </div>
                <div id="error" class="error-box"></div>
                <div id="tokenCalculation" class="calculation-result"></div>
                <div id="result"></div>
            </div>

            <script>
                // Show API key instructions and calculate button when OpenAI is selected
                document.getElementById('modelSelect').addEventListener('change', function() {
                    const apiKeyInfo = document.getElementById('apiKeyInfo');
                    const calculateButton = document.getElementById('calculateButton');
                    
                    if (this.value === 'openai') {
                        apiKeyInfo.style.display = 'block';
                        calculateButton.style.display = 'block';
                    } else {
                        apiKeyInfo.style.display = 'none';
                        calculateButton.style.display = 'none';
                    }
                });
                
                // API Key test functionality
                document.getElementById('testApiKeyButton').addEventListener('click', async function() {
                    const resultElement = document.getElementById('apiKeyTestResult');
                    const loading = document.getElementById('loading');
                    const error = document.getElementById('error');
                    
                    loading.style.display = 'block';
                    loading.textContent = 'Testing OpenAI API key... Please wait...';
                    resultElement.style.display = 'none';
                    error.style.display = 'none';
                    
                    try {
                        const response = await fetch('/test-openai-key');
                        const data = await response.json();
                        
                        console.log("API key test response:", data);
                        
                        let resultHtml = '';
                        if (data.status === 'success') {
                            resultHtml = `<div class="api-test-success">
                                <strong>✅ Success!</strong> Your OpenAI API key is valid and working correctly.
                            </div>`;
                        } else {
                            let errorDetails = '';
                            if (data.details) {
                                errorDetails = `<br><br><strong>Error details:</strong> ${data.details}`;
                            }
                            
                            resultHtml = `<div class="api-test-error">
                                <strong>❌ Error:</strong> ${data.message}${errorDetails}
                            </div>`;
                            
                            if (data.message.includes('Rate limit')) {
                                resultHtml += `<div style="margin-top: 10px;">
                                    <strong>Note:</strong> Rate limits might be temporary. Try again in a few minutes.
                                </div>`;
                            }
                        }
                        
                        resultElement.innerHTML = resultHtml;
                        resultElement.style.display = 'block';
                    } catch (err) {
                        console.error("API key test error:", err);
                        resultElement.innerHTML = `<div class="api-test-error">
                            <strong>❌ Error:</strong> Could not complete the API key test. Error: ${err.message}
                        </div>`;
                        resultElement.style.display = 'block';
                    } finally {
                        loading.style.display = 'none';
                    }
                });
                
                // Token calculation functionality
                document.getElementById('calculateButton').addEventListener('click', async function() {
                    const submissionFile = document.getElementById('submissionFile').files[0];
                    const idealFile = document.getElementById('idealFile').files[0];
                    
                    if (!submissionFile || !idealFile) {
                        alert('Please select both submission and ideal answer ZIP files.');
                        return;
                    }
                    
                    const loading = document.getElementById('loading');
                    const tokenCalculation = document.getElementById('tokenCalculation');
                    const error = document.getElementById('error');
                    
                    loading.style.display = 'block';
                    loading.textContent = 'Calculating token usage... Please wait...';
                    tokenCalculation.style.display = 'none';
                    error.style.display = 'none';
                    
                    try {
                        const formData = new FormData();
                        formData.append('submission', submissionFile);
                        formData.append('ideal', idealFile);
                        
                        const response = await fetch('/calculate-tokens', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            // Display calculation results
                            let resultHtml = `<h3>OpenAI Usage Estimation</h3>
                                            <p>Based on the submitted files, here's an estimate of OpenAI API usage:</p>
                                            
                                            <div class="infobox">
                                                <strong>Files Analysis:</strong>
                                                <ul>
                                                    <li>Total functions: ${data.total_functions}</li>
                                                    <li>Ideal answer functions: ${data.ideal_functions}</li>
                                                    <li>Submission functions: ${data.submission_functions}</li>
                                                </ul>
                                            </div>
                                            
                                            <div class="infobox">
                                                <strong>Token Usage:</strong>
                                                <ul>
                                                    <li>Total estimated tokens: ${data.total_tokens.toLocaleString()}</li>
                                                    <li>API requests needed: ${data.total_requests.toLocaleString()}</li>
                                                    <li>Estimated cost: ${data.estimated_cost_usd}</li>
                                                    <li>Token counting method: ${data.tokenizer_used}</li>
                                                </ul>
                                            </div>`;
                                            
                            // Add rate limit information and warnings
                            resultHtml += `
                                <div class="infobox">
                                    <strong>OpenAI Rate Limits:</strong>
                                    <table>
                                        <tr>
                                            <th>Limit Type</th>
                                            <th>Your Usage</th>
                                            <th>OpenAI Limit</th>
                                            <th>Status</th>
                                        </tr>
                                        <tr>
                                            <td>Tokens Per Minute (TPM)</td>
                                            <td>${data.total_tokens.toLocaleString()}</td>
                                            <td>${data.tokens_per_minute_limit.toLocaleString()}</td>
                                            <td class="${data.will_hit_tpm_limit ? 'warning' : 'safe'}">${data.will_hit_tpm_limit ? '⚠️ Exceeds limit' : '✅ Within limit'}</td>
                                        </tr>
                                        <tr>
                                            <td>Requests Per Minute (RPM)</td>
                                            <td>${data.total_requests.toLocaleString()}</td>
                                            <td>${data.requests_per_minute_limit.toLocaleString()}</td>
                                            <td class="${data.will_hit_rpm_limit ? 'warning' : 'safe'}">${data.will_hit_rpm_limit ? '⚠️ Exceeds limit' : '✅ Within limit'}</td>
                                        </tr>
                                        <tr>
                                            <td>Requests Per Day (RPD)</td>
                                            <td>${data.total_requests.toLocaleString()}</td>
                                            <td>${data.requests_per_day_limit.toLocaleString()}</td>
                                            <td class="${data.will_hit_daily_limit ? 'warning' : 'safe'}">${data.will_hit_daily_limit ? '⚠️ Exceeds limit' : '✅ Within limit'}</td>
                                        </tr>
                                    </table>
                                </div>`;
                                
                            // Add processing time and recommendations
                            resultHtml += `
                                <div class="infobox">
                                    <strong>Processing Time Estimate:</strong>
                                    <p>With the OpenAI rate limits, this would take approximately:</p>
                                    <ul>
                                        <li><strong>${data.minutes_required}</strong> minute(s) due to rate limiting</li>
                                        <li><strong>${data.days_required}</strong> day(s) of your daily request quota</li>
                                    </ul>
                                    
                                    ${(data.will_hit_rpm_limit || data.will_hit_tpm_limit || data.will_hit_daily_limit) ? 
                                    `<p class="warning">⚠️ This assignment will be affected by OpenAI rate limits.</p>
                                    <p><strong>Recommendations:</strong></p>
                                    <ul>
                                        <li>Consider using the Ollama model for faster processing without rate limits</li>
                                        <li>Be patient as the processing will take longer due to automatic retries</li>
                                        <li>For frequent large assignments, consider upgrading your OpenAI account</li>
                                    </ul>` : 
                                    `<p class="safe">✅ This assignment should process without hitting OpenAI rate limits.</p>`}
                                </div>`;
                                
                            tokenCalculation.innerHTML = resultHtml;
                            tokenCalculation.style.display = 'block';
                        } else {
                            error.innerHTML = `<p><strong>Error:</strong> ${data.detail}</p>`;
                            error.style.display = 'block';
                        }
                    } catch (err) {
                        error.innerHTML = `<p><strong>Error:</strong> ${err.message}</p>`;
                        error.style.display = 'block';
                    } finally {
                        loading.style.display = 'none';
                    }
                });
                
                document.getElementById('uploadForm').onsubmit = async (e) => {
                    e.preventDefault();
                    
                    const formData = new FormData(e.target);
                    
                    const loading = document.getElementById('loading');
                    const result = document.getElementById('result');
                    const error = document.getElementById('error');
                    const tokenCalculation = document.getElementById('tokenCalculation');
                    
                    loading.style.display = 'block';
                    loading.textContent = 'Evaluating submission... Please wait...';
                    result.style.display = 'none';
                    error.style.display = 'none';
                    tokenCalculation.style.display = 'none';
                    
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
                                            <p><strong>Functions Matched:</strong> ${data.matched_functions}</p>`;
                            
                            // Add token usage information if OpenAI was used
                            console.log("Response data:", data); // For debugging
                            if (data.token_usage) {
                                const tokenCount = typeof data.token_usage.total_tokens === 'number' 
                                    ? data.token_usage.total_tokens.toLocaleString() 
                                    : data.token_usage.total_tokens;
                                
                                resultHtml += `
                                <div class="token-usage">
                                    <h4 style="margin-top: 0;">OpenAI API Usage</h4>
                                    <p><strong>Total Tokens Used:</strong> ${tokenCount}</p>
                                    <p><strong>Estimated Cost:</strong> ${data.token_usage.estimated_cost_usd}</p>
                                </div>`;
                            }
                            
                            resultHtml += `<h4>Detailed Report:</h4><ul>`;
                            
                            for (const [func, details] of Object.entries(data.report)) {
                                const color = details.status === 'Correct' ? 'green' : 
                                            details.status === 'Missing' ? 'red' : 'orange';
                                resultHtml += `<li><strong>${func}:</strong> 
                                             <span style="color: ${color}">${details.status}</span>
                                             (Similarity: ${details.similarity * 100}%)</li>`;
                            }
                            resultHtml += '</ul>';
                            result.innerHTML = resultHtml;
                            result.style.display = 'block';
                        } else {
                            console.error("API Error:", data); // Log detailed error
                            let errorMessage = data.detail || "Unknown error occurred";
                            
                            // Add more context to API key errors
                            if (errorMessage.includes('OpenAI API key')) {
                                errorMessage += `<br><br>Please check:<br>
                                    1. Your API key is correctly set in the .env file<br>
                                    2. The key hasn't expired or been revoked<br>
                                    3. You have billing enabled on your OpenAI account`;
                            }
                            
                            // Add more context to rate limit errors
                            if (errorMessage.includes('rate limit')) {
                                errorMessage += `<br><br>Even though your token usage calculation was within limits, you might have other processes using your API key, or OpenAI might have temporarily reduced your rate limits.`;
                            }
                            
                            error.innerHTML = `<p><strong>Error:</strong> ${errorMessage}</p>`;
                            error.style.display = 'block';
                            
                            // Show specific instructions for API key errors
                            if (data.detail && data.detail.includes('OpenAI API key')) {
                                document.getElementById('apiKeyInfo').style.display = 'block';
                            }
                        }
                    } catch (err) {
                        error.innerHTML = `<p><strong>Error:</strong> ${err.message}</p>`;
                        error.style.display = 'block';
                    } finally {
                        loading.style.display = 'none';
                    }
                };
            </script>
        </body>
    </html>
    """
    return html_content

@app.get("/test-openai-key")
async def test_openai_key():
    """Test if the OpenAI API key is valid."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_api_key_here":
        return {
            "status": "error",
            "message": "OpenAI API key not configured. Please update the OPENAI_API_KEY in your .env file.",
            "key_provided": False
        }
    
    try:
        # Simple test request to OpenAI
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use a tiny amount of text to minimize token usage
        test_text = "Hello world"
        
        logger.info(f"Testing OpenAI API key: {OPENAI_API_KEY[:5]}{'*' * 10}")
        
        response = requests.post(
            OPENAI_API_URL,
            headers=headers,
            json={
                "model": "text-embedding-ada-002",
                "input": test_text
            },
            timeout=10
        )
        
        # Log response information
        logger.info(f"Test response status: {response.status_code}")
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "OpenAI API key is valid.",
                "response_code": response.status_code
            }
        elif response.status_code == 401:
            return {
                "status": "error",
                "message": "Invalid or expired OpenAI API key.",
                "response_code": response.status_code,
                "details": response.text
            }
        elif response.status_code == 429:
            return {
                "status": "error",
                "message": "Rate limit exceeded. Your API key is valid but you've hit OpenAI's rate limits.",
                "response_code": response.status_code,
                "details": response.text,
                "headers": dict(response.headers)
            }
        else:
            return {
                "status": "error",
                "message": f"OpenAI API returned an unexpected status code: {response.status_code}",
                "response_code": response.status_code,
                "details": response.text
            }
            
    except Exception as e:
        logger.error(f"Error testing OpenAI API key: {e}")
        return {
            "status": "error",
            "message": f"Error testing OpenAI API key: {str(e)}",
            "exception": str(type(e).__name__)
        }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port) 