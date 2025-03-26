"""
Test script to demonstrate token management in the AI Assignment Checker.
"""
import os
import sys
import json
from utils.tokenizer_utils import count_tokens, truncate_by_token_limit, safe_truncate_code
from prompts import FEEDBACK_PROMPT_TEMPLATE

def generate_large_code_sample(size_kb=100):
    """Generate a large code sample for testing."""
    base_code = """
def process_data(data):
    results = []
    for item in data:
        processed = item * 2
        results.append(processed)
    return results

# Large data processing function
"""
    # Repeat to create large code
    repeated = base_code
    while len(repeated.encode('utf-8')) < size_kb * 1024:
        repeated += f"""
def function_{len(repeated)}():
    print("This is a test function")
    # Adding more lines
    # to create large code samples
    x = 1
    y = 2
    z = x + y
    return z
"""
    return repeated

def test_token_management():
    print("Testing token management utilities")
    
    # Generate large code samples
    print("\n1. Generating test code...")
    student_code = generate_large_code_sample(size_kb=50)  # 50KB of code
    ideal_code = generate_large_code_sample(size_kb=30)    # 30KB of code
    
    # Count tokens
    print("\n2. Counting tokens in original code...")
    student_tokens = count_tokens(student_code)
    ideal_tokens = count_tokens(ideal_code)
    print(f"Student code: {student_tokens} tokens ({len(student_code)} chars)")
    print(f"Ideal code: {ideal_tokens} tokens ({len(ideal_code)} chars)")
    
    # Test code truncation
    print("\n3. Testing code truncation...")
    max_tokens = 5000
    truncated_student = safe_truncate_code(student_code, max_tokens)
    truncated_student_tokens = count_tokens(truncated_student)
    print(f"Truncated student code: {truncated_student_tokens} tokens")
    print(f"Truncation successful: {truncated_student_tokens <= max_tokens}")
    
    # Test prompt construction
    print("\n4. Testing prompt construction with token limits...")
    # Create dummy structure analysis
    structure_analysis = {
        "variables": {"missing_variables": ["x", "y"], "extra_variables": ["z"]},
        "control_flow": {"missing_control_structures": [], "extra_control_structures": []},
        "function_calls": {"missing_calls": ["print"], "extra_calls": []}
    }
    
    # Create dummy similar contexts
    similar_contexts = [
        {"function_name": f"func_{i}", "code": generate_large_code_sample(size_kb=10), "similarity": 0.8 - (i * 0.1)}
        for i in range(5)  # 5 similar contexts of 10KB each
    ]
    
    # Format similar contexts as separate chunks
    context_snippets = [
        f"Function: {ctx.get('function_name', 'Unknown')}\nSimilarity: {ctx.get('similarity', 0):.2f}\nCode:\n{ctx.get('code', 'No code available')}"
        for ctx in similar_contexts
    ]
    
    # Calculate base prompt tokens
    base_prompt = FEEDBACK_PROMPT_TEMPLATE.format(
        student_code=safe_truncate_code(student_code, 15000),
        ideal_code=safe_truncate_code(ideal_code, 15000),
        structure_diff=json.dumps(structure_analysis, indent=2),
        similar_contexts="__PLACEHOLDER__"
    )
    
    base_tokens = count_tokens(base_prompt.replace("__PLACEHOLDER__", ""))
    print(f"Base prompt (without contexts): {base_tokens} tokens")
    
    # Calculate available tokens for similar contexts
    max_prompt_tokens = 100000
    max_response_tokens = 20000
    available_tokens = max_prompt_tokens - max_response_tokens - base_tokens
    print(f"Available tokens for contexts: {available_tokens}")
    
    # Truncate similar contexts
    truncated_contexts = truncate_by_token_limit(context_snippets, available_tokens)
    contexts_tokens = count_tokens(truncated_contexts)
    print(f"Truncated contexts: {contexts_tokens} tokens")
    
    # Final prompt
    final_prompt = FEEDBACK_PROMPT_TEMPLATE.format(
        student_code=safe_truncate_code(student_code, 15000),
        ideal_code=safe_truncate_code(ideal_code, 15000),
        structure_diff=json.dumps(structure_analysis, indent=2),
        similar_contexts=truncated_contexts
    )
    
    final_tokens = count_tokens(final_prompt)
    print(f"Final prompt: {final_tokens} tokens")
    print(f"Within limit: {final_tokens <= max_prompt_tokens - max_response_tokens}")
    
    return {
        "student_tokens": student_tokens,
        "ideal_tokens": ideal_tokens,
        "truncated_student_tokens": truncated_student_tokens,
        "base_tokens": base_tokens,
        "available_tokens": available_tokens,
        "contexts_tokens": contexts_tokens,
        "final_tokens": final_tokens,
        "is_within_limit": final_tokens <= max_prompt_tokens - max_response_tokens
    }

if __name__ == "__main__":
    results = test_token_management()
    print("\nSummary:")
    print(json.dumps(results, indent=2))
    
    if results["is_within_limit"]:
        print("\nToken management test successful! The prompt is within the token limit.")
        sys.exit(0)
    else:
        print("\nToken management test failed! The prompt exceeds the token limit.")
        sys.exit(1) 