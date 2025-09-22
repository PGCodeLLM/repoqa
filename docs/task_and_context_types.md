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
  - [Mixed Context](#mixed---dynamic-smart-allocation)
  - [Optimal Context](#optimal---fair-token-distribution)
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

Our modified version of RepoQA offers four context processing approaches, each optimized for different scenarios and token budgets:

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
- **Enhanced class method support**: Automatically includes parent class definitions for methods
- Preserves parameter lists, return types, and decorators
- Maintains repository structure overview
- Optimizes for maximum function discovery with proper class context

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

### `mixed` - Dynamic Smart Allocation

**Purpose**: Intelligently balances between signatures and complete implementations based on function size and available tokens.

**Characteristics**:
- **Includes**: Complete small functions + signatures of large functions
- **Best For**: General-purpose tasks requiring both breadth and depth
- **Function Count**: Optimized mix of complete and signature-only functions

**Processing Strategy**:
1. **Dynamic Token Allocation**: Distributes tokens based on function sizes rather than equal distribution
2. **Smart Prioritization**: Includes complete smaller functions that fit within token budget
3. **Fallback to Signatures**: Uses signature-only mode for functions too large to include completely
4. **Class Method Support**: Methods include parent class definitions for proper context
5. **Context Optimization**: Prioritizes functions closer to the needle (target function)

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

### `optimal` - Fair Token Distribution

**Purpose**: Provides fair token allocation across all functions with intelligent partial function inclusion for maximum context value.

**Characteristics**:
- **Includes**: Complete small functions + partial implementations of large functions
- **Best For**: Maximum information density while maintaining function order and fairness
- **Function Count**: All functions included with optimal token distribution
- **Token Efficiency**: Highest (95-100% token usage)

**Processing Strategy**:
1. **Fair Base Allocation**: Distributes token budget equally among all functions
2. **Spare Token Redistribution**: Unused tokens from small functions are redistributed to large functions
3. **Partial Function Support**: Large functions get meaningful partial implementations with truncation indicators
4. * Class Method Support**: Methods automatically include parent class definitions for proper context

**Algorithm**:
```python
base_budget_per_function = total_budget / num_functions

# Calculate spare tokens from small functions
spare_tokens = 0
for func in functions:
    if func_size <= base_budget_per_function:
        spare_tokens += (base_budget_per_function - func_size)

# Redistribute spare tokens to large functions
extra_tokens_per_large = spare_tokens / num_large_functions
for func in functions:
    if func_size <= base_budget_per_function:
        include_complete_function(func)
    else:
        available_tokens = base_budget_per_function + extra_tokens_per_large
        include_partial_function(func, available_tokens)
```

**Example Output**:
```python
# Path: src/config/validator.py
def parse_configuration(config_path: str) -> Dict[str, any]:
    """Parse configuration file and return settings."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        return {"error": "Configuration file not found"}
    # ... (truncated)

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
    # ... (truncated)
```

---

## 🏗️ Enhanced Class Method Support

All context processing types now include **enhanced class method support** that automatically provides proper class context for methods:

### Key Features

1. **Automatic Class Detection**: Uses tree-sitter AST parsing to detect when functions are methods within classes
2. **Parent Class Context**: Methods automatically include their parent class definition for proper understanding
3. **Proper Indentation**: Method signatures are correctly indented within their class context
4. **Cross-Context Consistency**: Class method enhancement works across all context types (body, signature, mixed, optimal)

### Example Enhancement

**Before Enhancement** (methods without class context):
```python
def validate_user(self, user_data: Dict) -> bool:
def save_user(self, user: User) -> None:
```

**After Enhancement** (methods with class context):
```python
class UserManager:
    def validate_user(self, user_data: Dict) -> bool:
    def save_user(self, user: User) -> None:
```

### Benefits

- **Better Context Understanding**: Models can better understand method relationships and class structure
- **Improved Code Comprehension**: Class context helps models understand the purpose and scope of methods
- **Enhanced Signature Clarity**: Method signatures are more meaningful when shown with their class context
- **Consistent Behavior**: All context types provide the same enhanced class method support

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

## � Technical Implementation Details

### Enhanced Function Extraction

The RepoQA system uses advanced tree-sitter AST parsing to extract functions with enhanced context awareness:

```python
def _extract_functions_from_content(language: str, content: str):
    """Extract all functions with class context for methods."""
    # Uses tree-sitter to parse code into AST
    # Detects parent classes for methods
    # Creates enhanced signatures with class context
    # Preserves original function boundaries and metadata
```

### Key Implementation Features

1. **Tree-sitter AST Parsing**: Uses language-specific parsers for accurate code structure analysis
2. **Parent Class Detection**: Automatically traverses AST to find parent class nodes for methods
3. **Enhanced Signature Generation**: Creates context-aware signatures that include class definitions
4. **Metadata Preservation**: Maintains function boundaries, byte positions, and classification (method vs function)

### Context Processing Pipeline

```python
# 1. Extract all functions with enhanced context
functions = _extract_functions_from_content(language, content)

# 2. Apply context-specific processing
if context_type == "signature":
    # Group methods by class to avoid duplicate class definitions
    # Create optimized signature representations
elif context_type == "mixed":
    # Binary decision: complete functions OR signatures only
    # Enhanced class method handling in both modes
elif context_type == "optimal":
    # Fair token distribution with partial function support
    # Class method enhancement for both complete and partial functions
```

### Token Efficiency Optimizations

- **Smart Class Grouping**: In signature context, methods are grouped by class to avoid duplicate class definitions
- **Enhanced Token Allocation**: Class context is included in token calculations for accurate budgeting
- **Consistent Enhancement**: All context types benefit from the same class method enhancement logic

---

## �📊 Performance Characteristics

### Token Efficiency Comparison

| Context Type | Typical Token Usage | Function Coverage | Implementation Depth |
|--------------|-------------------|------------------|---------------------|
| **Body**     | 90-100%          | Low (2-5 functions) | Complete implementations |
| **Signature** | 50-70%           | High (10-20 functions) | Signatures only |
| **Mixed**    | 85-100%          | Medium (5-12 functions) | Mixed (complete + signatures) |
| **Optimal**  | 95-100%          | High (all functions) | Fair partial + complete |

### Task Type Suitability

| Task Type | Best Context Type | Reasoning |
|-----------|------------------|-----------|
| **needle_search** | `optimal` or `mixed` or `body` | Needs function implementations to understand behavior |
| **echo_signature** | `signature` or `optimal` | Only needs signatures, maximize coverage |  
| **find_file** | `signature` or `optimal` | File navigation benefits from broad repository view |

### Computational Performance

- **Processing Time**: `signature` < `mixed` < `optimal` < `body`
- **Memory Usage**: `signature` < `mixed` < `optimal` < `body`
- **Token Generation**: `signature` (lowest) < `mixed` < `optimal` < `body` (highest)

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

3. **Use `mixed` when**:
   - You want balanced coverage and depth
   - Token budget is moderate (16k-32k tokens)
   - General-purpose evaluation

4. **Use `optimal` when**:
   - You want maximum information density
   - You need fair representation of all functions
   - You want consistent function ordering across context types
   - You want enhanced class method support with proper context
   - Token budget allows for meaningful partial functions (recommended default)

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
