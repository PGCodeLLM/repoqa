# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

from tree_sitter_languages import get_parser

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

FUNCTION_QUERY = {
    "python": "(function_definition name: (_)) @fdef",
    "java": "(method_declaration name: (_)) @fdef",
    "typescript": "(function_declaration name: (_)) @fdef",
    "rust": "(function_item name: (_)) @fdef",
    "cpp": "(function_definition declarator: (function_declarator declarator: (identifier))) @fdef",
    "go": "(function_declaration name: (_)) @fdef",
}

COMMENT_QUERY = {
    "python": [
        "(block (expression_statement (string) @docstring))",
        "(comment) @comment",
    ],
    "java": ["(line_comment) @comment", "(block_comment) @comment"],
    "cpp": ["(comment) @comment"],
    "rust": ["(line_comment) @comment", "(block_comment) @comment"],
    "typescript": ["(comment) @comment"],
    "go": ["(comment) @comment"],
}

FUNCTION_NAME_QUERY = {
    "python": """
        ((function_definition
          name: (identifier) @function_name))
    """,
    "java": """
        (method_declaration
          name: (identifier) @method_name)
    """,
    "typescript": """
        (function_declaration
          name: (identifier) @function_name)
    """,
    "rust": """
        (function_item
          name: (identifier) @function_name)
    """,
    "cpp": """
        (function_definition
          name: (identifier) @function_name)
    """,
}


def topological_sort(graph):
    # Stack to store the topological order
    stack = []
    # Set to keep track of visited nodes
    visited = set()

    # Recursive function to process nodes
    def dfs(node):
        # Mark the current node as visited
        visited.add(node)
        # Recurse for all the vertices adjacent to this vertex
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                dfs(neighbour)
        # Push current vertex to stack which stores the result
        stack.append(node)

    # Call the recursive helper function to store the topological sort starting from all vertices one by one
    for node in graph:
        if node not in visited:
            dfs(node)

    return stack


def progress(note: str = "processing"):
    return Progress(
        TextColumn(f"{note} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )


def extract_function_signature(language: str, file_content: str, start_byte: int, end_byte: int) -> str:
    """
    Extracts the function signature from file_content between start_byte and end_byte using Tree-sitter.
    Supports: python, java, typescript, rust, cpp, go.
    """
    parser = get_parser(language)
    source_bytes = bytes(file_content, "utf8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    # Language-specific node types for function definitions
    node_types = {
        "python": ["function_definition", "async_function_definition"],
        "java": ["method_declaration", "constructor_declaration"],
        "typescript": ["function_declaration", "method_definition"],
        "rust": ["function_item"],
        "cpp": ["function_definition"],
        "go": ["function_declaration", "method_declaration"],
    }
    types = node_types.get(language, [])

    def is_within(node):
        # Allow some tolerance for whitespace/comments
        return node.start_byte <= start_byte and node.end_byte >= end_byte

    def find_signature_node(node):
        if node.type in types and is_within(node):
            return node
        for child in node.children:
            result = find_signature_node(child)
            if result:
                return result
        return None

    func_node = find_signature_node(root)
    if not func_node:
        return ""

    # Find the first child that starts the function body (':' for Python, '{' for others)
    sig_end = func_node.end_byte  # fallback
    for child in func_node.children:
        # For Python, function body starts with ':'
        if language == "python" and child.type == ":":
            sig_end = child.end_byte
            break
        # For other languages, function body starts with '{'
        if language != "python" and child.type == "{":
            sig_end = child.start_byte
            break

    sig_text = source_bytes[func_node.start_byte:sig_end].decode("utf8")
    return sig_text.strip()


def extract_all_function_signatures(language: str, file_content: str) -> str:
    """
    Extracts all function signatures from file_content using Tree-sitter.
    Returns a string containing all function signatures separated by newlines.
    Supports: python, java, typescript, rust, cpp, go.
    """
    parser = get_parser(language)
    source_bytes = bytes(file_content, "utf8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    # Language-specific node types for function definitions
    node_types = {
        "python": ["function_definition", "async_function_definition"],
        "java": ["method_declaration", "constructor_declaration"],
        "typescript": ["function_declaration", "method_definition"],
        "rust": ["function_item"],
        "cpp": ["function_definition"],
        "go": ["function_declaration", "method_declaration"],
    }
    types = node_types.get(language, [])

    def find_function_nodes(node):
        functions = []
        if node.type in types:
            functions.append(node)
        for child in node.children:
            functions.extend(find_function_nodes(child))
        return functions

    function_nodes = find_function_nodes(root)
    signatures = []

    for func_node in function_nodes:
        # Find the first child that starts the function body (':' for Python, '{' for others)
        sig_end = func_node.end_byte  # fallback
        for child in func_node.children:
            # For Python, function body starts with ':'
            if language == "python" and child.type == ":":
                sig_end = child.end_byte
                break
            # For other languages, function body starts with '{'
            if language != "python" and child.type == "{":
                sig_end = child.start_byte
                break

        sig_text = source_bytes[func_node.start_byte:sig_end].decode("utf8")
        signatures.append(sig_text.strip())

    return "\n".join(signatures)


def calculate_prompt_overhead_tokens(task_type: str, needle_name: str, needle_description: str, tokenizer) -> int:
    """Calculate the number of tokens used by prompt instructions and other non-code-context elements."""
    
    # Import here to avoid circular imports
    from repoqa.search_needle_function import INSTRUCTION, ECHO_SIGNATURE_INSTRUCTION, FIND_FILE_INSTRUCTION
    
    if task_type == "needle_search":
        # Build prompt template without code_context
        prompt_parts = [
            INSTRUCTION,  # instruction
            # code_context would go here but we skip it
            needle_description,  # description  
            INSTRUCTION  # instruction again
        ]
        prompt_without_context = "".join(prompt_parts)
        
    elif task_type == "echo_signature":
        prompt_without_context = (
            ECHO_SIGNATURE_INSTRUCTION
            + "\n" 
            + f"Function name: {needle_name}"
            + "\n"
            # code_context would go here but we skip it
            + "\n"
            + ECHO_SIGNATURE_INSTRUCTION
            + "\n"
            + f"Function name: {needle_name}"
        )
        
    elif task_type == "find_file":
        prompt_without_context = (
            FIND_FILE_INSTRUCTION
            + "\n"
            + f"Function name: {needle_name}"
            + "\n" 
            # code_context would go here but we skip it
            + "\n"
            + FIND_FILE_INSTRUCTION
            + "\n"
            + f"Function name: {needle_name}"
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return len(tokenizer.tokenize(prompt_without_context))


def make_task_id(lang: str, repo: str, needle_name: str) -> str:
    """Create a unique task identifier."""
    return f"{lang}::{repo}::{needle_name}"


def make_cache_id(lang: str, repo: str, needle_name: str, code_context_size: int, position_ratio: float) -> str:
    """Create a unique cache identifier."""
    return f"{lang}::{repo}::{needle_name}::{code_context_size}::{position_ratio}"


# Comment prefixes for different languages
COMMENT_PREFIX = {
    "python": "#",
    "java": "//",
    "typescript": "//",
    "rust": "//",
    "cpp": "//",
    "go": "//",
}


def extract_functions_from_content(language: str, content: str):
    """Extract all functions from content with their positions and full text, including class context for methods."""
    from tree_sitter_languages import get_language, get_parser
    
    parser = get_parser(language)
    source_bytes = bytes(content, "utf8")
    tree = parser.parse(source_bytes)
    
    # Helper function to find the parent class node
    def find_parent_class(function_node):
        """Find the parent class node for a function, if any."""
        current = function_node.parent
        while current:
            if current.type in ["class_definition", "class_declaration"]:
                return current
            current = current.parent
        return None
    
    # Helper function to extract class signature
    def extract_class_signature(class_node):
        """Extract class signature (class name and inheritance)."""
        class_text = source_bytes[class_node.start_byte:class_node.end_byte].decode("utf8")
        lines = class_text.split('\n')
        
        # For Python, find the line with ':'
        if language == "python":
            for i, line in enumerate(lines):
                if line.rstrip().endswith(':'):
                    return '\n'.join(lines[:i+1])
        else:
            # For other languages, find the line with '{'
            for i, line in enumerate(lines):
                if '{' in line:
                    return '\n'.join(lines[:i+1])
        
        # Fallback to first line if no clear separator found
        return lines[0] if lines else ""
    
    # Get function query for this language
    fn_query = get_language(language).query(FUNCTION_QUERY[language])
    functions = []
    
    for capture in fn_query.captures(tree.root_node):
        node, _ = capture
        function_text = source_bytes[node.start_byte:node.end_byte].decode("utf8")
        
        # Check if this function is inside a class
        parent_class = find_parent_class(node)
        is_method = parent_class is not None
        
        # Extract function signature (up to first { or :)
        lines = function_text.split('\n')
        signature_lines = []
        body_start_line = 0
        
        for i, line in enumerate(lines):
            signature_lines.append(line)
            # Check if this line ends the signature
            if language == "python" and line.rstrip().endswith(':'):
                body_start_line = i + 1
                break
            elif language != "python" and '{' in line:
                # For other languages, signature ends at the line with {
                body_start_line = i + 1
                break
        
        function_signature = '\n'.join(signature_lines)
        function_body = '\n'.join(lines[body_start_line:]) if body_start_line < len(lines) else ""
        
        # For methods, create enhanced signature with class context
        if is_method:
            class_signature = extract_class_signature(parent_class)
            # Add proper indentation to function signature to show it's inside a class
            indented_function_signature = '\n'.join(['    ' + line if line.strip() else line 
                                                   for line in function_signature.split('\n')])
            enhanced_signature = f"{class_signature}\n{indented_function_signature}"
            
            # For full text of methods, we might want to include minimal class context
            # But for now, keep the original function text to avoid token bloat
            enhanced_full_text = function_text
        else:
            # Regular function - no class context needed
            enhanced_signature = function_signature
            enhanced_full_text = function_text
        
        functions.append({
            'signature': enhanced_signature,
            'body': function_body,
            'full_text': enhanced_full_text,
            'start_byte': node.start_byte,
            'end_byte': node.end_byte,
            'start_line': content[:node.start_byte].count('\n'),
            'end_line': content[:node.end_byte].count('\n'),
            'is_method': is_method,
            'class_name': source_bytes[parent_class.start_byte:parent_class.end_byte].decode("utf8").split('\n')[0].strip() if parent_class else None
        })
    
    return functions


def _create_mixed_context(
    needle,
    file_content_list,
    position_ratio: float,
    code_context_size: int,
    language: str,
    tokenizer,
    repo_name: str
):
    """Create mixed context with dynamic allocation (complete small + signature large functions)."""
    
    needle_file_idx = next(i for i, (path, _) in enumerate(file_content_list) if path == needle["path"])
    
    # Step 1: Use existing logic to determine which files would be included
    # This respects position_ratio and gives us the same file selection as the original algorithm
    
    # Calculate prefix and suffix sizes
    ntoken_needle_estimate = len(tokenizer.tokenize(needle.get("name", "function")))
    prefix_size = int(code_context_size * position_ratio - ntoken_needle_estimate / 2)  
    suffix_size = code_context_size - ntoken_needle_estimate - prefix_size
    
    # Collect all functions from files that would be in the context, preserving order
    all_functions = []
    needle_function_info = None
    
    # Process files in order they would appear in context
    files_to_process = []
    
    # Add prefix files (in reverse order, then reverse the result)
    prefix_files = []
    temp_prefix_size = prefix_size
    for i in range(needle_file_idx - 1, -1, -1):
        if temp_prefix_size <= 0:
            break
        path, content = file_content_list[i]
        path_header = f"{COMMENT_PREFIX[language]} Path: {path}\n"
        path_tokens = len(tokenizer.tokenize(path_header))
        if temp_prefix_size - path_tokens > 0:
            prefix_files.insert(0, (path, content))  # Insert at beginning to maintain order
            temp_prefix_size -= path_tokens
        
    files_to_process.extend(prefix_files)
    
    # Add needle file
    files_to_process.append(file_content_list[needle_file_idx])
    
    # Add suffix files
    temp_suffix_size = suffix_size
    for i in range(needle_file_idx + 1, len(file_content_list)):
        if temp_suffix_size <= 0:
            break
        path, content = file_content_list[i]  
        path_header = f"{COMMENT_PREFIX[language]} Path: {path}\n"
        path_tokens = len(tokenizer.tokenize(path_header))
        if temp_suffix_size - path_tokens > 0:
            files_to_process.append((path, content))
            temp_suffix_size -= path_tokens
    
    # Step 2: Extract all functions from these files
    for path, content in files_to_process:
        functions = extract_functions_from_content(language, content)
        for func in functions:
            func['path'] = path
            func['is_needle'] = False
            
            # Check if this is the needle function
            if path == needle["path"] and func['start_byte'] <= needle["start_byte"] < func['end_byte']:
                func['is_needle'] = True
                needle_function_info = func
            
            all_functions.append(func)
    
    # Ensure needle function is found
    if not needle_function_info:
        # Fallback: create needle function info manually
        needle_path, needle_content = file_content_list[needle_file_idx]
        needle_code = needle_content[needle["start_byte"] : needle["end_byte"]]
        return {
            'code_context': needle_code,
            'needle_token_start': 0, 
            'needle_token_end': len(tokenizer.tokenize(needle_code)),
            'code_context_ntokens': len(tokenizer.tokenize(needle_code))
        }
    
    # Step 3: Calculate optimal tokens per function
    if len(all_functions) == 0:
        return {
            'code_context': "",
            'needle_token_start': 0,
            'needle_token_end': 0,
            'code_context_ntokens': 0
        }
    
    # Calculate overhead for path headers
    overhead_tokens = 0
    unique_paths = list(dict.fromkeys(func['path'] for func in all_functions))  # Preserve order
    for path in unique_paths:
        path_header = f"{COMMENT_PREFIX[language]} Path: {path}\n"
        overhead_tokens += len(tokenizer.tokenize(path_header))
    
    available_tokens = code_context_size - overhead_tokens
    
    # Step 4: Build context with dynamic token allocation
    # Instead of rigid per-function allocation, use greedy approach with redistribution
    context_parts = []
    current_path = None
    needle_token_start = 0
    needle_token_end = 0
    
    # First pass: calculate what each function needs and wants
    function_requirements = []
    for func in all_functions:
        signature_tokens = len(tokenizer.tokenize(func['signature']))
        
        # Calculate ideal tokens (signature + full body)
        if func['body'].strip():
            full_function = func['signature'] + '\n' + func['body']
            ideal_tokens = len(tokenizer.tokenize(full_function))
        else:
            ideal_tokens = signature_tokens
            
        function_requirements.append({
            'function': func,
            'signature_tokens': signature_tokens,
            'ideal_tokens': ideal_tokens,
            'allocated_tokens': 0,
            'content': ''
        })
    
    # Second pass: allocate tokens with balanced approach
    functions_too_large = []
    remaining_budget = available_tokens
    
    # Sort functions to prioritize needle function first, then by size (smaller first for better fit)
    sorted_requirements = sorted(function_requirements, 
                                key=lambda x: (not x['function']['is_needle'], x['ideal_tokens']))
    
    # First ensure all functions get at least their signature if possible
    signatures_total = sum(req['signature_tokens'] for req in function_requirements)
    if signatures_total > available_tokens:
        # Can't fit all signatures, prioritize needle
        for req in sorted_requirements:
            func = req['function']
            if req['signature_tokens'] <= remaining_budget:
                req['allocated_tokens'] = req['signature_tokens']
                req['content'] = func['signature']
                remaining_budget -= req['signature_tokens']
            elif func['is_needle']:
                # Always include needle signature, even if it exceeds budget
                req['allocated_tokens'] = req['signature_tokens']
                req['content'] = func['signature']
                remaining_budget = 0
                functions_too_large.append(func)
                break
    else:
        # All signatures can fit, allocate signatures first
        for req in function_requirements:
            req['allocated_tokens'] = req['signature_tokens']
            req['content'] = req['function']['signature']
            remaining_budget -= req['signature_tokens']
        
        # Then determine which functions can fit completely with remaining tokens
        # Mixed context uses binary decision: complete function OR signature only
        body_candidates = [req for req in sorted_requirements if req['function']['body'].strip()]
        
        for req in body_candidates:
            func = req['function']
            additional_needed = req['ideal_tokens'] - req['allocated_tokens']
            
            if additional_needed > 0 and remaining_budget >= additional_needed:
                # Only allocate if we can fit the COMPLETE function body
                req['allocated_tokens'] += additional_needed
                remaining_budget -= additional_needed
                
                # Include complete function - for methods use enhanced signature + body, for functions use full text
                if func['is_method']:
                    req['content'] = func['signature'] + '\n' + func['body']
                else:
                    req['content'] = func['full_text']
            # else keep just signature (already set above)
    
    # Warn about functions that are too large
    if functions_too_large:
        needle_info = f"Function: {needle.get('name', 'unknown')} in {needle.get('path', 'unknown file')}"
        task_info = f"Repo: {repo_name}"
        print(f"⚠️  Warning: {needle_info} ({task_info})")
        print(f"   {len(functions_too_large)} function(s) have signatures exceeding available budget.")
    
    # Third pass: build final context from allocated content
    current_class = None
    for req in function_requirements:
        func = req['function']
        
        if not req['content']:  # Function was skipped
            continue
            
        # Add path header if this is a new file
        if func['path'] != current_path:
            if current_path is not None:
                context_parts.append('\n')  # Blank line before new path block
            path_header = f"{COMMENT_PREFIX[language]} Path: {func['path']}\n"
            context_parts.append(path_header)
            current_path = func['path']
            current_class = None  # Reset class context when changing files
        
        # Calculate tokens used so far (for needle position tracking)
        tokens_so_far = len(tokenizer.tokenize(''.join(context_parts)))
        
        # Track needle position
        if func['is_needle']:
            needle_token_start = tokens_so_far
            needle_token_end = tokens_so_far + len(tokenizer.tokenize(req['content']))
        
        # Handle class grouping for methods
        if func['is_method']:
            # Check if we need to add class definition
            if func['class_name'] != current_class:
                # New class - add class definition
                class_def = func['signature'].split('\n')[0]  # First line is class definition
                context_parts.append(class_def + '\n')
                current_class = func['class_name']
            
            # For methods, use content without class definition (already added above)
            if '\n' in req['content'] and req['content'].split('\n')[0].strip().startswith('class '):
                # Remove class definition line from method content
                method_content = '\n'.join(req['content'].split('\n')[1:])
                context_parts.append(method_content)
            else:
                context_parts.append(req['content'])
        else:
            # Standalone function - add as-is and reset class context
            context_parts.append(req['content'])
            current_class = None
        
        # Add proper spacing between functions
        if not req['content'].endswith('\n'):
            context_parts.append('\n')
        context_parts.append('\n')  # Extra newline for spacing between functions
    
    final_context = ''.join(context_parts).rstrip()
    total_tokens = len(tokenizer.tokenize(final_context))
    
    return {
        'code_context': final_context,
        'needle_token_start': needle_token_start,
        'needle_token_end': needle_token_end,
        'code_context_ntokens': total_tokens
    }


def _create_optimal_context(
    needle,
    file_content_list,
    position_ratio: float,
    code_context_size: int,
    language: str,
    tokenizer,
    repo_name: str
):
    """Create optimal context with fair token distribution and partial function support."""
    
    needle_file_idx = next(i for i, (path, _) in enumerate(file_content_list) if path == needle["path"])
    
    # Step 1: Determine which files to include (same logic as mixed context)
    ntoken_needle_estimate = len(tokenizer.tokenize(needle.get("name", "function")))
    prefix_size = int(code_context_size * position_ratio - ntoken_needle_estimate / 2)  
    suffix_size = code_context_size - ntoken_needle_estimate - prefix_size
    
    # Collect all functions from files that would be in the context, preserving order
    all_functions = []
    needle_function_info = None
    
    # Process files in order they would appear in context
    files_to_process = []
    
    # Add prefix files (in reverse order, then reverse the result)
    prefix_files = []
    temp_prefix_size = prefix_size
    for i in range(needle_file_idx - 1, -1, -1):
        if temp_prefix_size <= 0:
            break
        path, content = file_content_list[i]
        path_header = f"{COMMENT_PREFIX[language]} Path: {path}\n"
        path_tokens = len(tokenizer.tokenize(path_header))
        if temp_prefix_size - path_tokens > 0:
            prefix_files.insert(0, (path, content))  # Insert at beginning to maintain order
            temp_prefix_size -= path_tokens
        
    files_to_process.extend(prefix_files)
    
    # Add needle file
    files_to_process.append(file_content_list[needle_file_idx])
    
    # Add suffix files
    temp_suffix_size = suffix_size
    for i in range(needle_file_idx + 1, len(file_content_list)):
        if temp_suffix_size <= 0:
            break
        path, content = file_content_list[i]  
        path_header = f"{COMMENT_PREFIX[language]} Path: {path}\n"
        path_tokens = len(tokenizer.tokenize(path_header))
        if temp_suffix_size - path_tokens > 0:
            files_to_process.append((path, content))
            temp_suffix_size -= path_tokens
    
    # Step 2: Extract all functions from these files
    for path, content in files_to_process:
        functions = extract_functions_from_content(language, content)
        for func in functions:
            func['path'] = path
            func['is_needle'] = False
            
            # Check if this is the needle function
            if path == needle["path"] and func['start_byte'] <= needle["start_byte"] < func['end_byte']:
                func['is_needle'] = True
                needle_function_info = func
            
            all_functions.append(func)
    
    # Ensure needle function is found
    if not needle_function_info:
        # Fallback: create needle function info manually
        needle_path, needle_content = file_content_list[needle_file_idx]
        needle_code = needle_content[needle["start_byte"] : needle["end_byte"]]
        return {
            'code_context': needle_code,
            'needle_token_start': 0, 
            'needle_token_end': len(tokenizer.tokenize(needle_code)),
            'code_context_ntokens': len(tokenizer.tokenize(needle_code))
        }
    
    # Step 3: Calculate overhead for path headers
    if len(all_functions) == 0:
        return {
            'code_context': "",
            'needle_token_start': 0,
            'needle_token_end': 0,
            'code_context_ntokens': 0
        }
    
    overhead_tokens = 0
    unique_paths = list(dict.fromkeys(func['path'] for func in all_functions))  # Preserve order
    for path in unique_paths:
        path_header = f"{COMMENT_PREFIX[language]} Path: {path}\n"
        overhead_tokens += len(tokenizer.tokenize(path_header))
    
    available_tokens = code_context_size - overhead_tokens
    
    # Step 4: Fair token distribution algorithm
    n_functions = len(all_functions)
    base_budget_per_function = available_tokens // n_functions if n_functions > 0 else 0
    
    # Phase 1: Categorize functions and calculate spare tokens
    small_functions = []  # fit completely within base budget
    large_functions = []  # need more than base budget
    spare_tokens = 0
    
    for func in all_functions:
        signature_tokens = len(tokenizer.tokenize(func['signature']))
        full_tokens = len(tokenizer.tokenize(func['full_text']))
        
        func['signature_tokens'] = signature_tokens
        func['full_tokens'] = full_tokens
        
        if full_tokens <= base_budget_per_function:
            small_functions.append(func)
            spare_tokens += (base_budget_per_function - full_tokens)
        else:
            large_functions.append(func)
    
    # Phase 2: Distribute spare tokens equally among large functions
    if len(large_functions) > 0:
        extra_tokens_per_large = spare_tokens // len(large_functions)
        tokens_per_large_function = base_budget_per_function + extra_tokens_per_large
    else:
        tokens_per_large_function = 0
    
    # Step 5: Build context with allocated tokens IN ORIGINAL ORDER
    context_parts = []
    current_path = None
    current_class = None
    needle_token_start = 0
    needle_token_end = 0
    
    for func in all_functions:  # Maintain original order!
        # Add path header if this is a new file
        if func['path'] != current_path:
            if current_path is not None:
                context_parts.append('\n')  # Blank line before new path block
            path_header = f"{COMMENT_PREFIX[language]} Path: {func['path']}\n"
            context_parts.append(path_header)
            current_path = func['path']
            current_class = None  # Reset class context when changing files
        
        # Calculate tokens used so far (for needle position tracking)
        tokens_so_far = len(tokenizer.tokenize(''.join(context_parts)))
        
        # Generate content based on function category
        if func in small_functions:
            # Include complete function, but for methods, use enhanced signature + body
            if func['is_method']:
                # For small methods, show class context + complete method
                function_content = func['signature'] + '\n' + func['body']
            else:
                # For standalone functions, use full text as usual
                function_content = func['full_text']
        else:
            # Large function - use allocated tokens with partial content
            available_tokens = tokens_per_large_function
            
            if available_tokens <= func['signature_tokens']:
                # Not enough tokens even for signature - just include signature
                function_content = func['signature']
            else:
                # Include signature + partial body with truncation indicator
                available_for_body = available_tokens - func['signature_tokens']
                
                # Reserve tokens for truncation indicator
                truncation_indicator = "    # ... (truncated)"
                truncation_tokens = len(tokenizer.tokenize(truncation_indicator))
                available_for_body -= truncation_tokens
                
                if available_for_body > 0 and func['body'].strip():
                    # Include partial body
                    body_lines = func['body'].split('\n')
                    included_body = ""
                    current_tokens = 0
                    
                    for line in body_lines:
                        line_with_newline = line + '\n'
                        line_tokens = len(tokenizer.tokenize(line_with_newline))
                        
                        if current_tokens + line_tokens <= available_for_body:
                            included_body += line_with_newline
                            current_tokens += line_tokens
                        else:
                            break
                    
                    # Construct partial function
                    function_content = func['signature'] + '\n'
                    if included_body.strip():
                        function_content += included_body.rstrip() + '\n' + truncation_indicator
                    else:
                        function_content = func['signature']  # No room for body
                else:
                    # Just signature
                    function_content = func['signature']
        
        # Track needle position
        if func['is_needle']:
            needle_token_start = tokens_so_far
            needle_token_end = tokens_so_far + len(tokenizer.tokenize(function_content))
        
        # Handle class grouping for methods
        if func['is_method']:
            # Check if we need to add class definition
            if func['class_name'] != current_class:
                # New class - add class definition
                class_def = func['signature'].split('\n')[0]  # First line is class definition
                context_parts.append(class_def + '\n')
                current_class = func['class_name']
            
            # For methods, use content without class definition (already added above)
            if '\n' in function_content and function_content.split('\n')[0].strip().startswith('class '):
                # Remove class definition line from method content
                method_content = '\n'.join(function_content.split('\n')[1:])
                context_parts.append(method_content)
            else:
                context_parts.append(function_content)
        else:
            # Standalone function - add as-is and reset class context
            context_parts.append(function_content)
            current_class = None
        
        # Add proper spacing between functions
        if not function_content.endswith('\n'):
            context_parts.append('\n')
        context_parts.append('\n')  # Extra newline for spacing between functions
    
    final_context = ''.join(context_parts).rstrip()
    total_tokens = len(tokenizer.tokenize(final_context))
    
    return {
        'code_context': final_context,
        'needle_token_start': needle_token_start,
        'needle_token_end': needle_token_end,
        'code_context_ntokens': total_tokens
    }


def clean_context_comments(
    language: str,
    prefix: str,
    needle_code: str,
    suffix: str,
    tokenizer,
    context_paths: str,
    top_prefix_file: str,
    bot_suffix_file: str,
    position_ratio: float,
    add_padding: bool,
):
    """Clean comments from context while maintaining positioning."""
    # Import locally to avoid circular import
    from repoqa.search_needle_function import clean_partial_file, clean_segment_comments
    
    prefix_orig_size = len(tokenizer.tokenize(prefix))
    needle_orig_size = len(tokenizer.tokenize(needle_code))
    suffix_orig_size = len(tokenizer.tokenize(suffix))

    # If there is are prefix files, it might get chopped off preventing proper parsing
    # we fully parse the top prefix file to avoid errors
    if top_prefix_file:
        second_path = f"{COMMENT_PREFIX[language]} Path: {context_paths[1]}"
        prefix_lines = prefix.split("\n")
        top_file_lines = 0
        lines_after_target = []
        target_found = False
        for line in prefix_lines:
            if target_found:
                lines_after_target.append(line)
            elif second_path in line:
                target_found = True
                lines_after_target.append(line)
            else:
                top_file_lines += 1
        top_file_cleaned = clean_partial_file(
            language, top_prefix_file, top_file_lines, context_paths[0]
        )
        rest_files_cleaned = clean_segment_comments(
            language, "\n".join(lines_after_target), context_paths
        )
        prefix_cleaned = top_file_cleaned + rest_files_cleaned
    else:
        prefix_cleaned = clean_segment_comments(language, prefix, context_paths)
    needle_cleaned = needle_code
    needle_cleaned = clean_segment_comments(language, needle_code, context_paths)

    # Same for suffix
    if bot_suffix_file:
        second_path = f"{COMMENT_PREFIX[language]} Path: {context_paths[-1]}"
        prefix_lines = prefix.split("\n")
        bot_file_lines = 0
        lines_before_target = []
        target_found = False
        for line in prefix_lines:
            if target_found:
                lines_before_target.append(line)
            elif second_path in line:
                target_found = True
                lines_before_target.append(line)
            else:
                bot_file_lines += 1
        top_file_cleaned = clean_partial_file(
            language, bot_suffix_file, bot_file_lines, context_paths[-1]
        )
        rest_files_cleaned = clean_segment_comments(
            language, "\n".join(lines_before_target), context_paths
        )
        suffix_cleaned = rest_files_cleaned + top_file_cleaned
    else:
        suffix_cleaned = clean_segment_comments(language, suffix, context_paths)

    if not add_padding:
        return prefix_cleaned, needle_cleaned, suffix_cleaned

    # Calculate amount of padding to prefix and suffix to maintain position
    prefix_clean_size = len(tokenizer.tokenize(prefix_cleaned))
    needle_clean_size = len(tokenizer.tokenize(needle_cleaned))
    suffix_clean_size = len(tokenizer.tokenize(suffix_cleaned))

    # Determine how much of needle padding go to prefix & suffix
    needle_tokens_removed = needle_orig_size - needle_clean_size
    needle_prefix_padding = int(needle_tokens_removed * position_ratio)
    needle_suffix_padding = needle_tokens_removed - needle_prefix_padding

    # Add more padding to compensate removal from prefix/suffix portions
    needle_prefix_padding = int(
        (needle_prefix_padding + prefix_orig_size - prefix_clean_size - 1)
    )
    needle_suffix_padding = int(
        (needle_suffix_padding + suffix_orig_size - suffix_clean_size - 1)
    )

    prefix_dummy = ""
    line = 0
    while needle_prefix_padding > 0:
        current = f"{COMMENT_PREFIX[language]} Line Number {line}\n"
        current_len = len(tokenizer.tokenize(current))
        needle_prefix_padding -= current_len
        prefix_dummy += current
        line += 1
    prefix_cleaned = prefix_dummy + "\n" + prefix_cleaned

    suffix_dummy = ""
    while needle_suffix_padding > 0:
        current = f"{COMMENT_PREFIX[language]} Line Number {line}\n"
        current_len = len(tokenizer.tokenize(current))
        needle_suffix_padding -= current_len
        line += 1
        suffix_dummy += current
    suffix_cleaned = suffix_cleaned + suffix_dummy + "\n"

    return prefix_cleaned, needle_cleaned, suffix_cleaned