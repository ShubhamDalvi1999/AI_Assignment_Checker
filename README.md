# AI Assignment Checker

A system that evaluates programming assignment submissions by comparing them to an ideal answer using AI-powered semantic analysis.

![example](https://github.com/user-attachments/assets/d5abd5eb-4b17-4df8-bb45-166abf702cb6)

## Prerequisites

- Python 3.8+
- MongoDB
- Ollama with Llama 3.2 3B model

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Ollama:
- Install Ollama from [ollama.ai](https://ollama.ai)
- Pull the Llama 3.2 3B model:
```bash
ollama pull llama3.2:3b
```
- Start Ollama server:
```bash
ollama serve
```

3. Set up MongoDB:
- Install MongoDB or run via Docker:
```bash
docker run -d -p 27017:27017 mongo
```

4. Run the application:
```bash
python app.py
```

5. Access the web interface at: http://localhost:8001

## Usage

1. Prepare two ZIP files:
   - Student submission containing Python files
   - Ideal answer containing Python files
2. Visit http://localhost:8001
3. Upload both ZIP files
4. Click "Evaluate" to get the comparison report

## Features

- Semantic code comparison using AI embeddings (using Llama 3.2 3B locally or OpenAI API)
- Function-level similarity analysis
- Simple web interface
- Performance optimized with local AI processing
- MongoDB vector storage for embeddings

## Notes

- The system currently supports Python files only
- Functions are matched by name between submission and ideal answer
- Similarity scores > 0.8 are considered "Correct" 
