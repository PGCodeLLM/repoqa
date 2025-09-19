# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from enum import Enum
from typing import List, Tuple

from transformers import AutoTokenizer
from tree_sitter_languages import get_language, get_parser

from repoqa.compute_score import compute_score, save_json
from repoqa.data import CACHE_DIR, get_repoqa_data
from repoqa.utility import COMMENT_QUERY, progress, topological_sort, extract_function_signature, extract_all_function_signatures, FUNCTION_QUERY

COMMENT_PREFIX = {
    "python": "#",
    "java": "//",
    "typescript": "//",
    "rust": "//",
    "cpp": "//",
    "go": "//",
}

# Model context below:
TEMPLATE = "instruction\ncode_context\ndescription\ninstruction"

INSTRUCTION = (
    "Based on the function description and code context,"
    " please retrieve and repeat the exact described function from the code context in a code block wrapped by ```:"
)

ECHO_SIGNATURE_INSTRUCTION = (
    "Based on the function description and code context," 
    " please retrieve and repeat the following function's signature from the code context in a code block wrapped by ```:"
)

ECHO_SIGNATURE_TEMPLATE = "instruction\nname\ncode_context\ninstruction\nname"

FIND_FILE_INSTRUCTION = (
    "Based on the function description and code context,"
    " output the file path where the following function is defined, without any additional text or explanation."
)

FIND_FILE_TEMPLATE = "instruction\nname\ncode_context\ninstruction\nname"

# Mode to clean context comments
class CleanComment(Enum):
    NoClean = "none"
    PositionalPadding = "positional_padding"
    NoPadding = "no_padding"


def _backward_tokenizable_lines(lines, tokenizer, max_tokens):
    """Return the text and tokens from bottom to top"""
    text = ""
    ntokens = 0
    is_break = False
    is_latest_line = True
    
    # For signature context, try to avoid breaking in the middle of function signatures
    in_function_signature = False
    
    for line in reversed(lines):
        # if the first processed line is not empty, we do not add a new line after it
        if is_latest_line:
            NEW_LINE = ''
            if line == '':
                NEW_LINE = '\n'
            is_latest_line = False
        else:
            NEW_LINE = '\n'

        new_ntokens = len(tokenizer.tokenize(line + NEW_LINE))
        
        # Check if we're in a function signature (simple heuristic)
        line_stripped = line.strip()
        starts_function = (line_stripped.startswith('def ') or 
                          line_stripped.startswith('class ') or
                          line_stripped.startswith('async def '))
        ends_function_sig = line_stripped.endswith(':') and (starts_function or in_function_signature)
        
        if starts_function:
            in_function_signature = True
        elif ends_function_sig:
            in_function_signature = False
        
        # If adding this line would exceed tokens, check if we should break
        if ntokens + new_ntokens > max_tokens:
            # If we're in the middle of a function signature, try to include the whole signature
            if in_function_signature and ntokens < max_tokens * 0.9:  # Allow 10% overflow for signatures
                # Include this line to complete the signature
                pass
            else:
                is_break = True
                break
                
        text = line + NEW_LINE + text
        ntokens += new_ntokens
        
    return text, ntokens, is_break


def _forward_tokenizable_lines(lines, tokenizer, max_tokens):
    """Return the text and tokens from top to bottom"""
    text = ""
    ntokens = 0
    is_break = False
    for line in lines:
        new_ntokens = len(tokenizer.tokenize(line + "\n"))
        if ntokens + new_ntokens > max_tokens:
            is_break = True
            break
        text += line + "\n"
        ntokens += new_ntokens
    if is_break:
        text = text + "...\n"
        ntokens += len(tokenizer.tokenize("...\n"))
    return text, ntokens, is_break


def filter_path_comments(capture, context_paths, source_bytes, comment_prefix):
    node, _ = capture
    text = source_bytes[node.start_byte : node.end_byte]
    for path in context_paths:
        if text.decode("utf8") == comment_prefix + " Path: " + path:
            return False
    return True


def clean_segment_comments(language, segment, context_paths):
    source_bytes = bytes(segment, "utf8")
    parser = get_parser(language)
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    # Remove comments from source code
    capture_list = []
    for query_str in COMMENT_QUERY[language]:
        comment_query = get_language(language).query(query_str)
        capture_list += comment_query.captures(root_node)

    # Filter out synethetic comments containing paths info
    filtered_capture = list(
        filter(
            lambda capture: filter_path_comments(
                capture, context_paths, source_bytes, COMMENT_PREFIX[language]
            ),
            capture_list,
        )
    )

    filtered_capture.sort(key=lambda cap: cap[0].start_byte, reverse=True)

    for node, _ in filtered_capture:
        source_bytes = source_bytes[: node.start_byte] + source_bytes[node.end_byte :]

    return source_bytes.decode("utf-8")


# Clean partial context due to context construction
def clean_partial_file(language, whole_file, partial_lines, path):
    path_comment = f"{COMMENT_PREFIX[language]} Path: {path}\n"
    source_bytes = bytes(whole_file, "utf8")
    parser = get_parser(language)
    tree = parser.parse(source_bytes)
    root_node = tree.root_node

    # Remove comments from source code
    capture_list = []
    for query_str in COMMENT_QUERY[language]:
        comment_query = get_language(language).query(query_str)
        capture_list += comment_query.captures(root_node)

    capture_list.sort(key=lambda cap: cap[0].start_byte, reverse=True)

    for node, _ in capture_list:
        new_line_count = source_bytes[node.start_byte : node.end_byte].count(b"\n")
        source_bytes = (
            source_bytes[: node.start_byte]
            + b"\n" * new_line_count
            + source_bytes[node.end_byte :]
        )

    return (
        path_comment
        + "\n".join(source_bytes.decode("utf-8").split("\n")[: partial_lines - 1])
        + "...\n"
    )


def _extract_functions_from_content(language: str, content: str):
    """Extract all functions from content with their positions and full text."""
    parser = get_parser(language)
    source_bytes = bytes(content, "utf8")
    tree = parser.parse(source_bytes)
    
    # Get function query for this language
    fn_query = get_language(language).query(FUNCTION_QUERY[language])
    functions = []
    
    for capture in fn_query.captures(tree.root_node):
        node, _ = capture
        function_text = source_bytes[node.start_byte:node.end_byte].decode("utf8")
        
        # Extract signature (up to first { or :)
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
        
        signature = '\n'.join(signature_lines)
        body = '\n'.join(lines[body_start_line:]) if body_start_line < len(lines) else ""
        
        functions.append({
            'signature': signature,
            'body': body,
            'full_text': function_text,
            'start_byte': node.start_byte,
            'end_byte': node.end_byte,
            'start_line': content[:node.start_byte].count('\n'),
            'end_line': content[:node.end_byte].count('\n')
        })
    
    return functions


def _create_optimal_context(
    needle,
    file_content_list: List[Tuple[str, str]],
    position_ratio: float,
    code_context_size: int,
    language: str,
    tokenizer,
    repo_name: str
):
    """Create optimal context where functions are truncated to fit within token limits."""
    
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
        functions = _extract_functions_from_content(language, content)
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
    total_tokens_used = 0
    
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
        
        # Then distribute remaining tokens for body content, prioritizing smaller functions first
        # (they're more likely to fit completely)
        body_candidates = [req for req in sorted_requirements if req['function']['body'].strip()]
        
        for req in body_candidates:
            func = req['function']
            additional_needed = req['ideal_tokens'] - req['allocated_tokens']
            
            if additional_needed > 0 and remaining_budget > 0:
                # Allocate as much as possible, but don't exceed what's needed or available
                additional_allocation = min(additional_needed, remaining_budget)
                req['allocated_tokens'] += additional_allocation
                remaining_budget -= additional_allocation
                
                # Build content with body
                available_for_body = req['allocated_tokens'] - req['signature_tokens']
                body_lines = func['body'].split('\n')
                body_part = ""
                body_tokens = 0
                
                for line in body_lines:
                    line_tokens = len(tokenizer.tokenize(line + '\n'))
                    if body_tokens + line_tokens <= available_for_body:
                        body_part += line + '\n'
                        body_tokens += line_tokens
                    else:
                        break
                
                if body_part.strip():
                    req['content'] = func['signature'] + '\n' + body_part.rstrip()
                # else keep just signature (already set above)
    
    # Warn about functions that are too large
    if functions_too_large:
        needle_info = f"Function: {needle.get('name', 'unknown')} in {needle.get('path', 'unknown file')}"
        task_info = f"Repo: {repo_name}"
        print(f"⚠️  Warning: {needle_info} ({task_info})")
        print(f"   {len(functions_too_large)} function(s) have signatures exceeding available budget.")
    
    # Third pass: build final context from allocated content
    for req in function_requirements:
        func = req['function']
        
        if not req['content']:  # Function was skipped
            continue
            
        # Add path header if this is a new file
        if func['path'] != current_path:
            if current_path is not None:
                context_parts.append('\n')
            path_header = f"{COMMENT_PREFIX[language]} Path: {func['path']}\n"
            context_parts.append(path_header)
            current_path = func['path']
        
        # Calculate tokens used so far (for needle position tracking)
        tokens_so_far = len(tokenizer.tokenize(''.join(context_parts)))
        
        # Track needle position
        if func['is_needle']:
            needle_token_start = tokens_so_far
            needle_token_end = tokens_so_far + len(tokenizer.tokenize(req['content']))
        
        context_parts.append(req['content'])
        # Add proper spacing between functions (like body context does)
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


def make_code_context(
    needle,
    file_content_list: List[Tuple[str, str]],
    position_ratio: float,
    code_context_size: int,
    language: str,
    clean_comments: CleanComment = CleanComment.NoClean,
    context_type: str = "body",
    repo_name: str = "unknown",
) -> str:
    """
    Slice the file_content_list such that:
    1. The slice contains code_context_size tokens
    2. The positon of the needle is at position_ratio of the slice*
    *May not be achievable if the needle is too close to the beginning or end of the file_content_list
    *May not be accurate as we will also insert file names at the beginning of each file
    *Token sizes might not be 100 accurate but should be close enough
    
    If context_type is "signature", uses function signatures instead of full file content.
    If context_type is "body" (default), uses full file content.
    If context_type is "optimal", finds optimal token allocation per function to fit within context_size.
    """
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    # Keep original file content list for needle extraction
    original_file_content_list = file_content_list

    needle_file_idx, needle_file_content = [
        (i, content)
        for i, (f, content) in enumerate(original_file_content_list)
        if f == needle["path"]
    ][0]

    # Handle optimal context type - entirely different logic
    if context_type == "optimal":
        return _create_optimal_context(
            needle, 
            file_content_list, 
            position_ratio, 
            code_context_size, 
            language, 
            tokenizer,
            repo_name
        )

    # For signature context type, convert file contents to function signatures and extract needle as signature
    if context_type == "signature":
        # First, extract all signatures from the needle file to find the needle function signature
        all_signatures = extract_all_function_signatures(language, needle_file_content)
        
        # Parse the file to find which function contains the needle start_byte
        parser = get_parser(language)
        source_bytes = bytes(needle_file_content, "utf8")
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
        
        def find_containing_function(node):
            if node.type in types and node.start_byte <= needle["start_byte"] < node.end_byte:
                return node
            for child in node.children:
                result = find_containing_function(child)
                if result:
                    return result
            return None
        
        needle_func_node = find_containing_function(root)
        if needle_func_node:
            # Extract just the signature part of this function
            sig_end = needle_func_node.end_byte  # fallback
            for child in needle_func_node.children:
                # For Python, function body starts with ':'
                if language == "python" and child.type == ":":
                    sig_end = child.end_byte
                    break
                # For other languages, function body starts with '{'
                if language != "python" and child.type == "{":
                    sig_end = child.start_byte
                    break
            needle_code = source_bytes[needle_func_node.start_byte:sig_end].decode("utf8").strip()
        else:
            # Fallback to original needle code if we can't find the function
            needle_code = needle_file_content[needle["start_byte"] : needle["end_byte"]]
        
        # Convert all file contents to function signatures
        processed_file_content_list = []
        for path, content in original_file_content_list:
            signatures = extract_all_function_signatures(language, content)
            processed_file_content_list.append((path, signatures))
        file_content_list = processed_file_content_list
        
        # Update needle_file_content to be the signatures for the needle file
        needle_file_content = file_content_list[needle_file_idx][1]
    else:
        # Extract needle code from original content for other task types
        needle_code = needle_file_content[needle["start_byte"] : needle["end_byte"]]
    
    # Check if all files fit in context window and warn if not
    total_tokens = 0
    for path, content in file_content_list:
        # Add tokens for path header
        path_header = f"{COMMENT_PREFIX[language]} Path: {path}\n"
        total_tokens += len(tokenizer.tokenize(path_header))
        
        # Add tokens for file content (respecting context_type)
        if context_type == "signature":
            content_to_tokenize = extract_all_function_signatures(language, content)
        elif context_type == "optimal":
            # For optimal, we don't warn here - warnings are handled in _create_optimal_context
            content_to_tokenize = ""
        else:
            content_to_tokenize = content
        
        total_tokens += len(tokenizer.tokenize(content_to_tokenize))
    
    # Warn if total exceeds context size (skip for optimal as it handles its own warnings)
    if total_tokens > code_context_size and context_type != "optimal":
        needle_info = f"Function: {needle.get('name', 'unknown')} in {needle.get('path', 'unknown file')}"
        task_info = f"Repo: {repo_name}"
        print(f"⚠️  Warning: {needle_info} ({task_info})")
        print(f"   All files contain {total_tokens} tokens, exceeding context size of {code_context_size}.")
        print(f"   Some files may be truncated in the final context.")
        if context_type == "body":
            print(f"   Consider using --context_type signature or optimal to reduce token usage.")

    ntoken_needle = len(tokenizer.tokenize(needle_code))

    # Used for if cleaning comments option is enabled (paths comments are skipped)
    context_paths = [needle["path"]]
    top_prefix_file = None
    bot_suffix_file = None

    prefix_size = int(code_context_size * position_ratio - ntoken_needle / 2)
    suffix_size = code_context_size - ntoken_needle - prefix_size

    # handling prefix of the needle file
    if context_type == "signature":
        # For signatures, we need to split the signatures string, not use byte positions
        # Find which signature contains our needle
        signatures = needle_file_content.split('\n')
        needle_sig_idx = -1
        for i, sig in enumerate(signatures):
            # Simple heuristic: if the signature contains the needle function name
            if needle.get('name', '') in sig:
                needle_sig_idx = i
                break
        
        if needle_sig_idx >= 0:
            prefix_signatures = signatures[:needle_sig_idx]
        else:
            prefix_signatures = []
            
        code_prefix, ntokens, is_break = _backward_tokenizable_lines(
            [COMMENT_PREFIX[language] + " Path: " + needle["path"]] + prefix_signatures,
            tokenizer,
            prefix_size,
        )
    else:
        # For body and optimal contexts, use the original logic
        code_prefix, ntokens, is_break = _backward_tokenizable_lines(
            [COMMENT_PREFIX[language] + " Path: " + needle["path"]]
            + needle_file_content[: needle["start_byte"]].split("\n"),
            tokenizer,
            prefix_size,
        )
    prefix_size -= ntokens

    # handling prefix of the previous files
    index = needle_file_idx - 1
    while not is_break and prefix_size > 0 and index >= 0:
        path, content = file_content_list[index]
        context_paths.insert(0, path)
        top_prefix_file = content
        prefix, ntokens, is_break = _forward_tokenizable_lines(
            [COMMENT_PREFIX[language] + " Path: " + path] + content.split("\n"),
            tokenizer,
            prefix_size,
        )
        code_prefix = prefix + code_prefix
        prefix_size -= ntokens
        index -= 1

    # handling suffix of the needle file
    if context_type == "signature":
        # For signatures, get signatures after the needle
        signatures = needle_file_content.split('\n')
        needle_sig_idx = -1
        for i, sig in enumerate(signatures):
            if needle.get('name', '') in sig:
                needle_sig_idx = i
                break
        
        if needle_sig_idx >= 0 and needle_sig_idx + 1 < len(signatures):
            suffix_signatures = signatures[needle_sig_idx + 1:]
        else:
            suffix_signatures = []
            
        code_suffix, ntokens, is_break = _forward_tokenizable_lines(
            suffix_signatures, tokenizer, suffix_size
        )
    else:
        # For body and optimal contexts, use the original logic
        code_suffix, ntokens, is_break = _forward_tokenizable_lines(
            needle_file_content[needle["end_byte"] :].split("\n"), tokenizer, suffix_size
        )
    suffix_size -= ntokens

    # handling suffix of the next files
    index = needle_file_idx + 1
    while not is_break and suffix_size > 0 and index < len(file_content_list):
        path, content = file_content_list[index]
        context_paths.append(path)
        bot_suffix_file = content
        suffix, ntokens, is_break = _forward_tokenizable_lines(
            [COMMENT_PREFIX[language] + " Path: " + path] + content.split("\n"),
            tokenizer,
            suffix_size,
        )
        code_suffix += suffix
        suffix_size -= ntokens
        index += 1

    # Remove the comments in code_prefix, needle_code, code_suffix and
    # pad the code_prefix and code_suffix to maintain the position of the needle
    if clean_comments != CleanComment.NoClean:
        code_prefix, needle_code, code_suffix = clean_context_comments(
            language,
            code_prefix,
            needle_code,
            code_suffix,
            tokenizer,
            context_paths,
            top_prefix_file,
            bot_suffix_file,
            position_ratio,
            clean_comments == CleanComment.PositionalPadding,
        )

    # For signature context, ensure proper spacing between parts
    if context_type == "signature":
        # Make sure each part ends with a newline for proper separation
        if code_prefix and not code_prefix.endswith('\n'):
            code_prefix += '\n'
        if needle_code and not needle_code.endswith('\n'):
            needle_code += '\n'
        if code_suffix and not code_suffix.endswith('\n'):
            code_suffix += '\n'
    
    code_context = code_prefix + needle_code + code_suffix

    needle_token_start = len(tokenizer.tokenize(code_prefix))
    needle_token_end = needle_token_start + len(tokenizer.tokenize(needle_code))
    code_context_ntokens = needle_token_end + len(tokenizer.tokenize(code_suffix))

    return {
        "code_context": code_context,
        "needle_token_start": needle_token_start,
        "needle_token_end": needle_token_end,
        "code_context_ntokens": code_context_ntokens,
    }


def make_task_id(lang, repo, needle_name):
    return f"{lang}::{repo}::{needle_name}"


def make_cache_id(lang, repo, needle_name, code_context_size, position_ratio):
    return f"{lang}::{repo}::{needle_name}::{code_context_size}::{position_ratio}"


def evaluate_model(
    model: str,
    base_url: str = None,
    backend: str = None,
    tensor_parallel_size: int = 1,
    code_context_size: int = 16 * 1024,
    max_new_tokens: int = 1024,
    result_dir: str = "results",
    languages: List[str] = None,
    caching: bool = True,  # if enabled, will cache the tasks which can be used to resume
    system_message: str = None,
    dataset_path: str = None,
    clean_ctx_comments: str = "none",
    eval_ignore_comments: bool = False,  # ignore comments during score computation
    trust_remote_code: bool = False,
    attn_implementation=None,
    task_type: str = "needle_search",
    context_type: str = "body",
):
    if backend is None:
        if base_url is not None:
            backend = "openai"
        else:
            backend = "vllm"
        print(f"Using {backend} as the backend")
    assert backend is not None, "Please specify the backend"

    if dataset_path is not None:
        with open(dataset_path) as f:
            dataset = json.load(f)
    else:
        dataset = get_repoqa_data()

    allowed_task_types = ["needle_search", "echo_signature", "find_file"]
    if task_type not in allowed_task_types:
        raise ValueError(
            f"Invalid task_type '{task_type}'. Allowed values: {allowed_task_types}"
        )

    # makedir if not exists
    os.makedirs(result_dir, exist_ok=True)
    context_size_dir = os.path.join(result_dir, f"ntoken_{code_context_size}")
    os.makedirs(context_size_dir, exist_ok=True)
    model_output_path = os.path.join(
        context_size_dir,
        f"{model.replace('/', '_slash_')}.jsonl",
    )

    # resume from model_output_file
    if os.path.exists(model_output_path):
        with open(model_output_path) as f:
            model_outputs = [json.loads(line) for line in f]
    else:
        model_outputs = []

    if clean_ctx_comments == "positional_padding":
        clean_ctx_comments = CleanComment.PositionalPadding
    elif clean_ctx_comments == "no_padding":
        clean_ctx_comments = CleanComment.NoPadding
    else:
        clean_ctx_comments = CleanComment.NoClean

    # resume tasks from cache if any
    # schema: {"cache_id": .., **task}
    extra = ""
    if clean_ctx_comments != CleanComment.NoClean:
        extra += "_clean_cmt"
    cache_file = os.path.join(
        CACHE_DIR, f"cache{extra}_ntoken_{code_context_size}_v1.jsonl"
    )
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache = {}
    if caching:
        print("🔥 Caching enabled")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cache = [json.loads(line) for line in f]
                cache = {c["cache_id"]: c for c in cache}
                # remove the cache_id field in c
                for c in cache.values():
                    c.pop("cache_id")
            print(f"Resuming from cache: {cache_file} with {len(cache)} tasks")

    resumed_task_ids = {
        make_task_id(r["language"], r["repo"], r["name"]) for r in model_outputs
    }

    # for each task we include
    # "repo", "name", "language", "path",
    # "template", "position_ratio", "description", "instruction", "code_context"
    # "needle_token_start", "needle_token_end", "code_context_ntokens"
    tasks = []
    for lang, repos in dataset.items():
        if languages is not None and lang not in languages:
            print(f"Skipping {lang} as it is not selected; selected: {languages}")
            continue

        print(f"🔥 Preparing code context for {lang}...")
        with progress(f"Processing {lang} context") as pbar:
            # !!!!!!!!!!! FOR TESTS ONLY !!!!!!!!!!!
            # for repo in pbar.track(repos):
            for repo in pbar.track(repos[:1]):
                # skip if the repo does not have needles
                if "needles" not in repo:
                    pbar.console.print(
                        f"⚠️ Skipping {repo['repo']} ({lang}) as it does not have `needles` -- do needle analysis first"
                    )
                    continue

                ordered_paths = topological_sort(repo["dependency"])
                file_content_list = [
                    (path, repo["content"][path]) for path in ordered_paths
                ]
                for i, needle in enumerate(repo["needles"]):
                    task_id = make_task_id(lang, repo["repo"], needle["name"])
                    if task_id in resumed_task_ids:
                        pbar.console.print(
                            f"Skipping {task_id} as it is already in the results"
                        )
                        continue

                    position_ratio = (i + 0.5) / len(repo["needles"])
                    cache_id = make_cache_id(
                        lang,
                        repo["repo"],
                        needle["name"],
                        code_context_size,
                        position_ratio,
                    )
                    if cache_id in cache:
                        tasks.append(cache[cache_id])
                        continue

                    # Get the file content for the needle's file
                    file_content = repo["content"][needle["path"]]
                    # Extract the function signature using the utility function
                    signature = extract_function_signature(
                        lang,
                        file_content,
                        needle["start_byte"],
                        needle["end_byte"],
                    )

                    task = {
                        "repo": repo["repo"],
                        "name": needle["name"],
                        "language": lang,
                        "path": needle["path"],
                        "position_ratio": position_ratio,
                        "description": f"\nFunction Description:{needle['description']}\n",
                        "instruction": INSTRUCTION,
                        "template": TEMPLATE,
                        "signature": signature,
                    }
                    code_context_info = make_code_context(
                        needle,
                        file_content_list,
                        position_ratio=position_ratio,
                        code_context_size=code_context_size,
                        language=lang,
                        clean_comments=clean_ctx_comments,
                        context_type=context_type,
                        repo_name=repo["repo"],
                    )
                    task.update(code_context_info)
                    tasks.append(task)

                    if caching:  # cache
                        with open(cache_file, "a") as f_out:
                            f_out.write(
                                json.dumps({"cache_id": cache_id, **task}) + "\n"
                            )
    # filter finished tasks again (in case a cache is used)
    tasks = [
        task
        for task in tasks
        if make_task_id(task["language"], task["repo"], task["name"])
        not in resumed_task_ids
    ]

    if len(tasks) == 0:
        print("No tasks to evaluate! Exiting...")
        return

    if backend == "openai":
        from repoqa.provider.openai import OpenAIProvider

        engine = OpenAIProvider(model, base_url=base_url)
    elif backend == "vllm":
        from repoqa.provider.vllm import VllmProvider

        engine = VllmProvider(
            model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=int(code_context_size * 1.5),  # Magic number
            trust_remote_code=trust_remote_code,
        )
    elif backend == "anthropic":
        from repoqa.provider.anthropic import AnthropicProvider

        engine = AnthropicProvider(model)
    elif backend == "hf":
        from repoqa.provider.hf import HfProvider

        engine = HfProvider(
            model,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
    elif backend == "google":
        from repoqa.provider.google import GoogleProvider

        engine = GoogleProvider(model)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if not system_message:
        print("🔥 System message is disabled")
        system_message = None

    with open(model_output_path, "a") as f_out:
        with progress(f"Running {model}") as pbar:
            for task in pbar.track(tasks):
                actual_position_ratio = (
                    task["needle_token_start"] / task["code_context_ntokens"]
                )
                pbar.console.print(
                    f"Searching {task['name']} in {task['repo']} ({task['language']}) -- "
                    f"position ratio: actual={actual_position_ratio:.2f}, expected={task['position_ratio']}"
                )
                prompt = ""
                if task_type == "needle_search":
                    for key in task["template"].split("\n"):
                        prompt += task[key]
                elif task_type == "echo_signature":
                    prompt = (
                        ECHO_SIGNATURE_INSTRUCTION
                        + "\n"
                        + f"Function name: {task['name']}"
                        + "\n"
                        + task["code_context"]
                        + "\n"
                        + ECHO_SIGNATURE_INSTRUCTION
                        + "\n"
                        + f"Function name: {task['name']}"
                    )
                elif task_type == "find_file":
                    prompt = (
                        FIND_FILE_INSTRUCTION
                        + "\n"
                        + f"Function name: {task['name']}"
                        + "\n"
                        + task["code_context"]
                        + "\n"
                        + FIND_FILE_INSTRUCTION
                        + "\n"
                        + f"Function name: {task['name']}"
                    )
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

                replies = engine.generate_reply(
                    prompt, n=1, max_tokens=max_new_tokens, system_msg=system_message
                )
                result = {**task, "output": replies, "task_type": task_type}
                # if task_type == "find_file":
                #     result["ground_truth_path"] = task["path"]
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
                model_outputs.append(result)

    file_base, _ = os.path.splitext(model_output_path)
    result_path = file_base + "-SCORES.json"
    output_json = compute_score(
        model,
        dataset,
        model_outputs,
        eval_ignore_comments or clean_ctx_comments != CleanComment.NoClean,
    )
    save_json(output_json, result_path)


def main():
    from fire import Fire

    Fire(evaluate_model)


if __name__ == "__main__":
    main()
