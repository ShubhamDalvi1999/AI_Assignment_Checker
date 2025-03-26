"""
Prompt templates for LLM-based feedback generation.
"""

FEEDBACK_PROMPT_TEMPLATE = """
You are an AI code reviewer. You are given:
1. The student's function implementation,
2. The ideal function implementation,
3. Structure differences (e.g., missing variables, control flows),
4. Similar functions from past codebase submissions.

Write constructive feedback for the student. Highlight:
- What they did right,
- What they missed,
- How they can improve,
- If needed, show examples from similar code.

### Student Code:
{student_code}

### Ideal Code:
{ideal_code}

### Structure Analysis:
{structure_diff}

### Similar Implementations:
{similar_contexts}

Respond with detailed feedback in 1–3 paragraphs.
""" 