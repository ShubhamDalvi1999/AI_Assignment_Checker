from typing import List, Dict, Any
import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import cosine
import os
import ast
import json
import logging
from datetime import datetime, UTC
from dotenv import load_dotenv
from prompts import FEEDBACK_PROMPT_TEMPLATE
import requests
from utils.tokenizer_utils import count_tokens, truncate_by_token_limit, safe_truncate_code

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token management constants
MAX_PROMPT_TOKENS = 100_000  # 100k tokens total for prompt
MAX_RESPONSE_TOKENS = 20_000  # Reserve 20k tokens for LLM output
MAX_CODE_TOKENS = 15_000     # Maximum tokens for individual code blocks

load_dotenv()

class RAGProcessor:
    def __init__(self, mongo_client: MongoClient):
        self.db = mongo_client[os.getenv("MONGODB_DB_NAME", "assignment_checker")]
        self.collection = self.db[os.getenv("MONGODB_COLLECTION_NAME", "embeddings")]
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.use_openai = os.getenv("USE_OPENAI", "False").lower() == "true"
        
    def store_code_context(self, function_name: str, code: str, embedding: np.ndarray, 
                          metadata: Dict[str, Any] = None) -> None:
        """Store code context with its embedding and metadata."""
        document = {
            "function_name": function_name,
            "code": code,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC)
        }
        self.collection.insert_one(document)
        
    def retrieve_similar_contexts(self, query_embedding: np.ndarray, 
                                top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve similar code contexts based on embedding similarity."""
        # Get all stored embeddings
        stored_docs = list(self.collection.find({}, {"embedding": 1, "code": 1, "metadata": 1, "function_name": 1}))
        
        # Calculate similarities
        similarities = []
        for doc in stored_docs:
            similarity = 1 - cosine(query_embedding, np.array(doc["embedding"]))
            doc["similarity"] = similarity
            similarities.append((similarity, doc))
            
        # Sort by similarity and get top k
        similarities.sort(reverse=True)
        return [doc for _, doc in similarities[:top_k]]
    
    def generate_feedback_with_openai(self, student_code: str, ideal_code: str, 
                                     structure_analysis: Dict, similar_contexts: List[Dict]) -> str:
        """Generate feedback using OpenAI's API with token management."""
        if not self.openai_api_key:
            logger.warning("OpenAI API key not set, skipping feedback generation")
            return "Feedback generation requires OpenAI API key."
        
        model_name = "gpt-4o"
        
        # Truncate large code blocks if needed 
        student_code = safe_truncate_code(student_code, MAX_CODE_TOKENS, model=model_name)
        ideal_code = safe_truncate_code(ideal_code, MAX_CODE_TOKENS, model=model_name)
        
        # Prepare formatted similar contexts as separate chunks
        context_snippets = [
            f"Function: {ctx.get('function_name', 'Unknown')}\nSimilarity: {ctx.get('similarity', 0):.2f}\nCode:\n{ctx.get('code', 'No code available')}"
            for ctx in similar_contexts if 'code' in ctx
        ]
        
        # Calculate tokens for the base prompt without similar contexts
        base_prompt = FEEDBACK_PROMPT_TEMPLATE.format(
            student_code=student_code,
            ideal_code=ideal_code,
            structure_diff=json.dumps(structure_analysis, indent=2),
            similar_contexts="__PLACEHOLDER__"
        )
        
        base_tokens = count_tokens(base_prompt.replace("__PLACEHOLDER__", ""), model=model_name)
        
        # Calculate available tokens for similar contexts
        available_tokens = MAX_PROMPT_TOKENS - MAX_RESPONSE_TOKENS - base_tokens
        logger.info(f"Base prompt tokens: {base_tokens}, Available for contexts: {available_tokens}")
        
        # Truncate similar contexts to fit remaining space
        similar_contexts_text = truncate_by_token_limit(context_snippets, available_tokens, model=model_name)
        
        # Final prompt with truncated similar contexts
        prompt = FEEDBACK_PROMPT_TEMPLATE.format(
            student_code=student_code,
            ideal_code=ideal_code,
            structure_diff=json.dumps(structure_analysis, indent=2),
            similar_contexts=similar_contexts_text or "No similar implementations found."
        )
        
        final_token_count = count_tokens(prompt, model=model_name)
        logger.info(f"Final prompt tokens: {final_token_count}")
        
        if final_token_count > MAX_PROMPT_TOKENS - MAX_RESPONSE_TOKENS:
            logger.warning(f"Prompt still exceeds target size: {final_token_count} tokens")
        
        # Call OpenAI API
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful programming assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": MAX_RESPONSE_TOKENS
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI feedback generation failed: {e}")
            return f"Could not generate feedback: {str(e)}"
    
    def generate_feedback_with_ollama(self, student_code: str, ideal_code: str, 
                                    structure_analysis: Dict, similar_contexts: List[Dict]) -> str:
        """Generate feedback using Ollama."""
        # Format similar contexts for prompt
        similar_contexts_text = "\n\n".join([
            f"Function: {ctx.get('function_name', 'Unknown')}\nSimilarity: {ctx.get('similarity', 0):.2f}\nCode:\n{ctx.get('code', 'No code available')}"
            for ctx in similar_contexts if 'code' in ctx
        ])
        
        # Format the prompt
        prompt = FEEDBACK_PROMPT_TEMPLATE.format(
            student_code=student_code,
            ideal_code=ideal_code,
            structure_diff=json.dumps(structure_analysis, indent=2),
            similar_contexts=similar_contexts_text or "No similar implementations found."
        )
        
        # Construct proper chat endpoint URL
        chat_endpoint_url = f"{self.ollama_api_url}/api/chat"
        
        # Call Ollama API
        try:
            response = requests.post(
                chat_endpoint_url,
                json={
                    "model": "llama3.2:3b",
                    "messages": [
                        {"role": "system", "content": "You are a helpful programming assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_length": 500,
                    "stream": False  # Disable streaming to get a single complete response
                }
            )
            response.raise_for_status()
            
            # Handle the JSON parsing carefully
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                # Try to extract the first valid JSON object from the response
                logger.warning(f"JSON parsing error: {e}. Attempting to extract first JSON object.")
                text = response.text.strip()
                try:
                    # Split by newlines and parse the first line as JSON
                    first_json = text.split('\n')[0]
                    result = json.loads(first_json)
                    logger.info("Successfully extracted first JSON object from response.")
                except (json.JSONDecodeError, IndexError):
                    # If that fails, return the raw response text
                    logger.error("Failed to extract JSON from response.")
                    return f"Failed to parse Ollama response. Raw response:\n{text[:500]}"
            
            # Handle different possible response formats
            if "message" in result and "content" in result["message"]:
                # Standard format: {"message": {"content": "..."}}
                return result["message"]["content"]
            elif "response" in result:
                # Alternative format: {"response": "...", "done": true}
                return result["response"]
            elif "content" in result:
                # Simpler format: {"content": "..."}
                return result["content"]
            else:
                # Unknown format, return the whole response as string
                logger.warning(f"Unknown Ollama response format: {result}")
                return f"Response format not recognized. Raw response: {json.dumps(result)}"
            
        except Exception as e:
            logger.error(f"Ollama feedback generation failed: {e}")
            return f"Could not generate feedback with Ollama: {str(e)}"
    
    def generate_feedback(self, student_code: str, ideal_code: str, 
                        structure_analysis: Dict, similar_contexts: List[Dict]) -> str:
        """Generate feedback using either OpenAI or Ollama based on configuration."""
        if self.use_openai and self.openai_api_key:
            return self.generate_feedback_with_openai(student_code, ideal_code, structure_analysis, similar_contexts)
        else:
            return self.generate_feedback_with_ollama(student_code, ideal_code, structure_analysis, similar_contexts)
    
    def generate_comparison_report(self, student_code: str, ideal_code: str, 
                                 student_embedding: np.ndarray, ideal_embedding: np.ndarray) -> Dict[str, Any]:
        """Generate a detailed comparison report using RAG."""
        # Retrieve similar contexts
        similar_contexts = self.retrieve_similar_contexts(student_embedding)
        
        # Calculate direct similarity
        direct_similarity = 1 - cosine(student_embedding, ideal_embedding)
        
        # Analyze code structure
        structure_analysis = self._analyze_code_structure(student_code, ideal_code)
        
        # Generate LLM-based feedback
        feedback = self.generate_feedback(
            student_code, 
            ideal_code, 
            structure_analysis, 
            similar_contexts
        )
        
        # Generate detailed report
        report = {
            "direct_similarity": float(direct_similarity),
            "similar_contexts": similar_contexts,
            "structure_analysis": structure_analysis,
            "recommendations": self._generate_recommendations(
                direct_similarity, 
                similar_contexts, 
                structure_analysis
            ),
            "feedback": feedback  # Add LLM-generated feedback
        }
        
        return report
    
    def _analyze_code_structure(self, student_code: str, ideal_code: str) -> Dict[str, Any]:
        """Analyze code structure differences."""
        # Parse both codes using AST
        student_tree = ast.parse(student_code)
        ideal_tree = ast.parse(ideal_code)
        
        # Compare structure
        structure_diff = {
            "variables": self._compare_variables(student_tree, ideal_tree),
            "control_flow": self._compare_control_flow(student_tree, ideal_tree),
            "function_calls": self._compare_function_calls(student_tree, ideal_tree)
        }
        
        return structure_diff
    
    def _compare_variables(self, student_tree: ast.AST, ideal_tree: ast.AST) -> Dict[str, Any]:
        """Compare variable usage between student and ideal code."""
        student_vars = set()
        ideal_vars = set()
        
        for node in ast.walk(student_tree):
            if isinstance(node, ast.Name):
                student_vars.add(node.id)
                
        for node in ast.walk(ideal_tree):
            if isinstance(node, ast.Name):
                ideal_vars.add(node.id)
                
        return {
            "student_variables": list(student_vars),
            "ideal_variables": list(ideal_vars),
            "missing_variables": list(ideal_vars - student_vars),
            "extra_variables": list(student_vars - ideal_vars)
        }
    
    def _compare_control_flow(self, student_tree: ast.AST, ideal_tree: ast.AST) -> Dict[str, Any]:
        """Compare control flow structures."""
        student_control = []
        ideal_control = []
        
        for node in ast.walk(student_tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                student_control.append(type(node).__name__)
                
        for node in ast.walk(ideal_tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                ideal_control.append(type(node).__name__)
                
        return {
            "student_control_flow": student_control,
            "ideal_control_flow": ideal_control,
            "missing_control_structures": list(set(ideal_control) - set(student_control)),
            "extra_control_structures": list(set(student_control) - set(ideal_control))
        }
    
    def _compare_function_calls(self, student_tree: ast.AST, ideal_tree: ast.AST) -> Dict[str, Any]:
        """Compare function calls between student and ideal code."""
        student_calls = set()
        ideal_calls = set()
        
        for node in ast.walk(student_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    student_calls.add(node.func.id)
                    
        for node in ast.walk(ideal_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    ideal_calls.add(node.func.id)
                    
        return {
            "student_calls": list(student_calls),
            "ideal_calls": list(ideal_calls),
            "missing_calls": list(ideal_calls - student_calls),
            "extra_calls": list(student_calls - ideal_calls)
        }
    
    def _generate_recommendations(self, similarity: float, 
                                similar_contexts: List[Dict[str, Any]], 
                                structure_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Similarity-based recommendations
        if similarity < self.similarity_threshold:
            recommendations.append("Consider reviewing similar implementations from the codebase")
            
        # Structure-based recommendations
        if structure_analysis["variables"]["missing_variables"]:
            recommendations.append(f"Consider using these variables: {', '.join(structure_analysis['variables']['missing_variables'])}")
            
        if structure_analysis["control_flow"]["missing_control_structures"]:
            recommendations.append(f"Consider using these control structures: {', '.join(structure_analysis['control_flow']['missing_control_structures'])}")
            
        if structure_analysis["function_calls"]["missing_calls"]:
            recommendations.append(f"Consider using these functions: {', '.join(structure_analysis['function_calls']['missing_calls'])}")
            
        return recommendations 