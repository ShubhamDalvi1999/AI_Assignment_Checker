# AI Assignment Checker - Architecture & Flow Documentation

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web Browser   │────▶│  FastAPI Server  │────▶│  File Handler   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                          │
                               │                          ▼
                               │                  ┌─────────────────┐
                               │                  │  Code Parser    │
                               │                  └─────────────────┘
                               │                          │
                               ▼                          ▼
┌─────────────────┐      ┌──────────────────┐       ┌─────────────────┐
│                 │      │ Vector Database  │       │   Embedding     │
│  MongoDB Atlas  │◀───▶│    (MongoDB)     │◀───▶ │   Generator     │
│                 │      │                  │       │                 │
└─────────────────┘      └──────────────────┘       └─────────────────┘
                               ▲      ▲                    ▲
                               │      │                    │
                               │      │                    │
┌─────────────────┐      ┌─────┴──────┴───────┐      ┌─────┴────────┐
│   Similarity    │◀───▶│  RAG Controller    │◀──▶ │  AI Models   │
│   Calculator    │      │                    │      │              │
└─────────────────┘      └────────────────────┘      │ ┌─────────┐  │
                               │      ▲              │ │ Ollama  │  │
                               │      │              │ │ Local   │  │
                               ▼      │              │ └─────────┘  │
┌─────────────────┐      ┌──────────────────┐        │ ┌─────────┐  │
│    Results      │◀────│    Evaluation    │◀────   │ │ OpenAI  │  │
│   Generator     │      │    Engine        │        │ │  API    │  │
└─────────────────┘      └──────────────────┘        └──────────────┘
```

## Component Description

### 1. Frontend Components
- **Web Browser Interface**
  - Simple HTML form for file uploads
  - JavaScript for async submission
  - Real-time result display
  - Model selection dropdown

### 2. Backend Components
- **FastAPI Server**
  - Handles HTTP requests
  - Manages file uploads
  - Coordinates processing flow
  - Error handling and responses

- **File Handler**
  - Processes ZIP file uploads
  - Creates temporary directories
  - Manages file cleanup
  - Extracts Python files

- **Code Parser**
  - Uses Python's AST module
  - Extracts function definitions
  - Maintains code structure
  - Returns function mappings

- **RAG Controller**
  - Coordinates retrieval and generation process
  - Manages embedding workflow
  - Orchestrates similarity comparisons
  - Handles model selection and routing

- **Vector Database (MongoDB)**
  - Stores and indexes embeddings
  - Enables semantic search
  - Manages vector similarity operations
  - Caches frequent comparisons

- **Embedding Generator**
  - Converts code to embeddings
  - Handles multiple model outputs
  - Normalizes embedding formats
  - Optimizes for storage

- **Similarity Calculator**
  - Computes cosine similarity
  - Handles vector comparisons
  - Normalizes similarity scores
  - Provides threshold checking

### 3. AI Components
- **AI Models**
  - Two embedding options:
    1. Ollama (Local)
       - Uses Llama 3.2 3B model
       - Runs on localhost:11434
       - No API key required
    2. OpenAI (Cloud)
       - Uses text-embedding-ada-002
       - Requires API key
       - Higher accuracy

### 4. Storage Components
- **MongoDB Atlas**
  - Stores embeddings
  - Manages vector data
  - Enables quick retrieval
  - Cloud-based storage

## Data Flow

1. **Upload Phase**
   ```
   User → ZIP Files → FastAPI → Temporary Storage
   ```

2. **Processing Phase**
   ```
   ZIP Files → Code Parser → Functions → AI Model → Embeddings
   ```

3. **Storage Phase**
   ```
   Embeddings → MongoDB → Vector Storage
   ```

4. **Comparison Phase**
   ```
   Student Code → Embeddings → Similarity Check → Results
   ```

5. **Response Phase**
   ```
   Results → JSON → Frontend → User Display
   ```

## Key Processes

### 1. File Upload Process
1. User selects two ZIP files
2. Files are uploaded to FastAPI server
3. Server creates temporary storage
4. Files are extracted and validated

### 2. Code Analysis Process
1. Python files are identified
2. AST parser extracts functions
3. Functions are mapped by name
4. Code structure is preserved

### 3. Embedding Generation
1. Model selection (Ollama/OpenAI)
2. Code text is processed
3. Embeddings are generated
4. Vectors are stored in MongoDB

### 4. Similarity Comparison
1. Function pairs are matched
2. Cosine similarity computed
3. Scores are calculated
4. Results are formatted

### 5. Result Generation
1. Overall score calculation
2. Function-level reporting
3. Status determination
4. JSON response creation

## Configuration

### Environment Variables (.env)
```
MONGODB_URI=<connection_string>
OPENAI_API_KEY=<api_key>
OLLAMA_API_URL=http://localhost:11434/api/embeddings
HOST=127.0.0.1
PORT=8001
```

### Model Selection
- **Ollama (Default)**
  - Local processing
  - No API key needed
  - Uses Llama 3.2 3B

- **OpenAI**
  - Cloud processing
  - API key required
  - Higher accuracy

## Error Handling

### 1. File Processing Errors
- ZIP file validation
- Python file extraction
- Function parsing

### 2. Model Errors
- Ollama server connection
- OpenAI API issues
- Model availability

### 3. Database Errors
- Connection issues
- Storage failures
- Retrieval problems

## Performance Considerations

1. **Memory Management**
   - Temporary file cleanup
   - Efficient embedding storage
   - Stream processing

2. **Processing Optimization**
   - Parallel embedding generation
   - Efficient similarity computation
   - Quick database operations

3. **Error Recovery**
   - Graceful failure handling
   - Clear error messages
   - User-friendly responses

## RAG Process Flow

### 1. Code Embedding Generation
```
Code → Parser → AI Model → Embedding Vector
```

### 2. Vector Storage and Indexing
```
Embedding → MongoDB Vector Store → Indexed Storage
```

### 3. Retrieval Process
```
Query → Vector Search → Similar Embeddings → Ranked Results
```

### 4. Comparison Workflow
```
Student Code → Embeddings → Vector Similarity → Scoring
```

## Technical Implementation Details

### 1. Embedding Process
```python
# Example embedding generation
def generate_embedding(code: str, model: EmbeddingModel) -> np.ndarray:
    if model == EmbeddingModel.OLLAMA:
        # Ollama local embedding
        vector = get_embedding_ollama(code)
    else:
        # OpenAI embedding
        vector = get_embedding_openai(code)
    return normalize_vector(vector)
```

### 2. Vector Storage
```python
# MongoDB vector storage
{
    "function_name": str,
    "embedding": np.ndarray,  # 1536-dim for OpenAI, variable for Ollama
    "metadata": {
        "source": str,
        "timestamp": datetime,
        "model_used": str
    },
    "is_ideal": bool
}
```

### 3. Similarity Computation
```python
# Cosine similarity calculation
def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return 1 - cosine(vec1, vec2)
```

## Performance Optimizations

### 1. Vector Operations
- Batch embedding generation
- Parallel processing of vectors
- Efficient similarity computations
- Vectorized operations

### 2. Database Optimizations
- Vector indexing for fast retrieval
- Caching frequent comparisons
- Batch database operations
- Query optimization

### 3. Memory Management
- Streaming large vectors
- Efficient vector storage
- Cache management
- Resource cleanup

## Error Handling and Recovery

### 1. Embedding Errors
- Model fallback options
- Retry mechanisms
- Error logging
- Graceful degradation

### 2. Vector Storage Errors
- Connection retry
- Data validation
- Backup strategies
- Recovery procedures

### 3. Comparison Errors
- Default scoring
- Error thresholds
- Result validation
- Fallback comparisons

## Monitoring and Logging

### 1. Performance Metrics
- Embedding generation time
- Vector storage latency
- Comparison speed
- Overall processing time

### 2. Error Tracking
- Model errors
- Storage errors
- Processing errors
- System status

### 3. Usage Statistics
- Model usage
- Storage utilization
- API calls
- Success rates 