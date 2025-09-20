# RepoQA Task Types and Context Processing Guide

This document provides a comprehensive guide to the **task types** and **context processing types** supported by the RepoQA benchmark for evaluating long-context code understanding.

## 📋 Table of Contents

- [Task Types](#-task-types)
  - [Needle Search](#needle_search---function-retrieval)
  - [Echo Signature](#echo_signature---signature-extraction)
  - [Find File](#find_file---file-path-location)
- [Context Processing Types](#-context-processing-types)
  - [Body Context](#body---complete-function-bodies)
  - [Signature Context](#signature---function-signatures-only)
  - [Optimal Context](#optimal---dynamic-smart-allocation)
- [Usage Examples](#-usage-examples)
- [Performance Characteristics](#-performance-characteristics)
- [Best Practices](#-best-practices)

---

## 🎯 Task Types

We modified RepoQA to support three distinct task types, each designed to evaluate different aspects of long-context code understanding:

### `needle_search` (original RepoQA) - Function Retrieval

**Purpose**: Evaluates the model's ability to locate and retrieve a specific function based on a natural language description, without revealing function names or obvious keywords.

**Task Description**: Given a large repository context and a natural language description of a function's behavior, the model must find and return the exact function implementation.

**Input**:
- Large code context (multiple files, sorted by dependency)
- Natural language description of target function behavior
- Instruction to retrieve the described function

**Expected Output**: Complete function implementation wrapped in code blocks (```).

**Example**:
    
````
Task: "Based on the function description and code context, retrieve and repeat the exact described function from the code context in a code block wrapped by ```:"

Description: "A function that validates configuration settings by checking if all required keys are present in the settings dictionary. The required keys are 'database', 'api', and 'logging'. Returns True if all keys exist, False otherwise."

Expected Output:
```python
def validate_settings(settings: Dict[str, any]) -> bool:
    """Validate configuration settings."""
    required_keys = ["database", "api", "logging"]
    return all(key in settings for key in required_keys)
```
````

**Evaluation**: Function similarity using AST-based comparison with configurable thresholds (default: 0.8).

---

### `echo_signature` - Signature Extraction

**Purpose**: Tests the model's ability to extract and reproduce function signatures accurately from complex codebases.

**Task Description**: Given a function name and code context, the model must locate the function and return only its signature (declaration line).

**Input**:
- Function name to locate
- Large code context containing the target function
- Instruction to return the signature

**Expected Output**: Function signature only, wrapped in code blocks.

**Example**:
````
Task: "Based on the function description and code context, please retrieve and repeat the following function's signature from the code context in a code block wrapped by ```:"

Function name: validate_settings
Code context: [large repository context]

Expected Output:
```python
def validate_settings(settings: Dict[str, any]) -> bool:
```
````

**Evaluation**: String similarity between extracted signature and ground truth signature.

---

### `find_file` - File Path Location

**Purpose**: Evaluates the model's ability to navigate repository structure and locate the file containing a specific function.

**Task Description**: Given a function name and repository context, the model must identify and return the file path where the function is defined.

**Input**:
- Function name to locate
- Large code context with multiple files
- Instruction to output the file path

**Expected Output**: File path only, no additional text.

**Example**:
```
Task: "Based on the function description and code context, output the file path where the following function is defined, without any additional text or explanation."

Function name: validate_settings
Code context: [repository with multiple files]

Expected Output:
src/config/validator.py
```

**Evaluation**: String match between predicted and actual file path.

---

## 🔧 Context Processing Types

Our modified version of RepoQA offers three context processing approaches, each optimized for different scenarios and token budgets:

### `body` (original RepoQA) - Complete Function Bodies

**Purpose**: Provides complete function implementations with full context.

**Characteristics**:
- **Includes**: Full function bodies, complete implementations, documentation
- **Best For**: Tasks requiring complete understanding of function logic
- **Function Count**: Fewer functions, but complete implementations

**Processing Strategy**:
- Prioritizes including complete function bodies
- Truncates at function boundaries to avoid partial implementations
- Preserves code structure and indentation
- Includes docstrings and inline comments

**Example Output**:
```python
# Path: src/config/validator.py
def validate_settings(settings: Dict[str, any]) -> bool:
    """Validate configuration settings."""
    required_keys = ["database", "api", "logging"]
    return all(key in settings for key in required_keys)

def get_database_url(settings: Dict[str, any]) -> Optional[str]:
    """Extract database URL from settings."""
    if "database" not in settings:
        return None
    db_config = settings["database"]
    if not all(k in db_config for k in ["host", "port", "name"]):
        return None
    return f"postgresql://{db_config['host']}:{db_config['port']}/{db_config['name']}"
```

---

### `signature` - Function Signatures Only

**Purpose**: Provides maximum function coverage by including only signatures and essential structure.

**Characteristics**:
- **Includes**: Function signatures, class definitions, key structural elements
- **Best For**: Tasks requiring broad understanding of available functions
- **Function Count**: Maximum coverage, signatures only

**Processing Strategy**:
- Extracts only function and class signatures using tree-sitter AST parsing
- Preserves parameter lists, return types, and decorators
- Maintains repository structure overview
- Optimizes for maximum function discovery

**Example Output**:
```python
# Path: src/config/validator.py
def parse_configuration(config_path: str) -> Dict[str, any]:
def validate_settings(settings: Dict[str, any]) -> bool:
def get_database_url(settings: Dict[str, any]) -> Optional[str]:
def initialize_logging(settings: Dict[str, any]) -> None:

# Path: src/database/connection.py
class DatabaseConnection:
def connect(self, url: str) -> bool:
def execute_query(self, query: str) -> List[Dict]:
def close_connection(self) -> None:
```

---

### `optimal` - Dynamic Smart Allocation

**Purpose**: Intelligently balances between signatures and complete implementations based on function size and available tokens.

**Characteristics**:
- **Includes**: Complete small functions + signatures of large functions
- **Best For**: General-purpose tasks requiring both breadth and depth
- **Function Count**: Optimized mix of complete and signature-only functions

**Processing Strategy**:
1. **Dynamic Token Allocation**: Distributes tokens based on function sizes rather than equal distribution
2. **Smart Prioritization**: Includes complete smaller functions that fit within token budget
3. **Fallback to Signatures**: Uses signature-only mode for functions too large to include completely
4. **Context Optimization**: Prioritizes functions closer to the needle (target function)

**Algorithm**:
```python
for each function in context:
    if (remaining_tokens >= function_size):
        include_complete_function()
        remaining_tokens -= function_size
    else:
        include_signature_only()
        remaining_tokens -= signature_size
```

**Example Output**:
```python
# Path: src/config/validator.py
def parse_configuration(config_path: str) -> Dict[str, any]:
    # Large function - signature only due to token constraints

def validate_settings(settings: Dict[str, any]) -> bool:
    """Validate configuration settings."""
    required_keys = ["database", "api", "logging"]
    return all(key in settings for key in required_keys)

def get_database_url(settings: Dict[str, any]) -> Optional[str]:
    """Extract database URL from settings."""
    if "database" not in settings:
        return None
    db_config = settings["database"]
    # ... complete implementation included
```

---

## 💡 Usage Examples

### Basic CLI Usage

```bash
# Needle search with body context (default)
repoqa.search_needle_function \
    --model "gpt-4o-mini" \
    --backend openai \
    --task_type needle_search \
    --context_type body \
    --code_context_size 16384

# Echo signature with signature context
repoqa.search_needle_function \
    --model "claude-3-5-sonnet-20241022" \
    --backend anthropic \
    --task_type echo_signature \
    --context_type signature \
    --code_context_size 32768

# Find file with optimal context
repoqa.search_needle_function \
    --model "gemini-1.5-pro-latest" \
    --backend google \
    --task_type find_file \
    --context_type optimal \
    --code_context_size 8192
```

### Advanced Configuration

```bash
# Multi-language evaluation with optimal context
repoqa.search_needle_function \
    --model "meta-llama/CodeLlama-34b-Instruct-hf" \
    --backend vllm \
    --task_type needle_search \
    --context_type optimal \
    --languages '["python", "java", "typescript"]' \
    --code_context_size 32768 \
    --tensor_parallel_size 4

# Custom dataset with signature context
repoqa.search_needle_function \
    --model "deepseek-ai/deepseek-coder-33b-instruct" \
    --backend vllm \
    --task_type echo_signature \
    --context_type signature \
    --dataset_path custom_dataset.json \
    --result_dir custom_results \
    --code_context_size 16384
```

### Programmatic Usage

```python
from repoqa.search_needle_function import evaluate_model

# Comprehensive evaluation across task types
evaluate_model(
    model="gpt-4o-mini",
    backend="openai",
    task_type="needle_search",
    context_type="optimal",
    code_context_size=16384,
    languages=["python", "java"],
    result_dir="evaluation_results"
)

# Signature extraction task
evaluate_model(
    model="claude-3-5-sonnet-20241022",
    backend="anthropic", 
    task_type="echo_signature",
    context_type="signature",
    code_context_size=32768
)
```

---

## 📊 Performance Characteristics

### Token Efficiency Comparison

| Context Type | Typical Token Usage | Function Coverage | Implementation Depth |
|--------------|-------------------|------------------|---------------------|
| **Body**     | 90-100%          | Low (2-5 functions) | Complete implementations |
| **Signature** | 50-70%           | High (10-20 functions) | Signatures only |
| **Optimal**  | 85-100%          | Medium (5-12 functions) | Mixed (complete + signatures) |

### Task Type Suitability

| Task Type | Best Context Type | Reasoning |
|-----------|------------------|-----------|
| **needle_search** | `optimal` or `body` | Needs function implementations to understand behavior |
| **echo_signature** | `signature` or `optimal` | Only needs signatures, maximize coverage |
| **find_file** | `signature` or `optimal` | File navigation benefits from broad repository view |

### Computational Performance

- **Processing Time**: `signature` < `optimal` < `body`
- **Memory Usage**: `signature` < `optimal` < `body`
- **Token Generation**: `signature` (lowest) < `optimal` < `body` (highest)

---

## 🎯 Best Practices

### Context Type Selection

1. **Use `body` when**:
   - You need complete function implementations
   - Token budget is generous (>32k tokens)
   - Task requires understanding complex logic

2. **Use `signature` when**:
   - You need maximum function coverage
   - Token budget is limited (<16k tokens)
   - Task focuses on function discovery/navigation

3. **Use `optimal` when**:
   - You want balanced coverage and depth
   - Token budget is moderate (16k-32k tokens)
   - General-purpose evaluation (recommended default)

### Task Type Selection

1. **Use `needle_search` for**:
   - Comprehensive code understanding evaluation
   - Testing semantic search capabilities
   - Real-world code retrieval scenarios

2. **Use `echo_signature` for**:
   - Signature extraction accuracy
   - API understanding evaluation
   - Function interface comprehension

3. **Use `find_file` for**:
   - Repository navigation skills
   - File organization understanding
   - Code location capabilities

### Token Budget Guidelines

```bash
# Conservative (8k-16k tokens)
--context_type signature --code_context_size 16384

# Balanced (16k-32k tokens) - RECOMMENDED
--context_type optimal --code_context_size 32768

# Comprehensive (32k+ tokens)
--context_type body --code_context_size 65536
```
