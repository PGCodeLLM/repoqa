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
from repoqa.utility import (
    COMMENT_QUERY, progress, topological_sort, extract_function_signature, 
    extract_all_function_signatures, FUNCTION_QUERY, calculate_prompt_overhead_tokens, 
    make_task_id, make_cache_id, COMMENT_PREFIX, extract_functions_from_content,
    _create_mixed_context, _create_optimal_context, clean_context_comments
)



# Model context below:
TEMPLATE = "instruction\ncode_context\ndescription\ninstruction"

INSTRUCTION = (
    "Based on the function description and code context,"
    " please retrieve and repeat the exact described function from the code context in a code block wrapped by ```:\n"
)

ECHO_SIGNATURE_INSTRUCTION = (
    "Based on the function description and code context," 
    " please retrieve and repeat the following function's signature from the code context in a code block wrapped by ```:\n"
)

ECHO_SIGNATURE_TEMPLATE = "instruction\nname\ncode_context\ninstruction\nname"

FIND_FILE_INSTRUCTION = (
    "Based on the function description and code context,"
    " output the file path where the following function is defined, without any additional text or explanation.\n"
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










def make_code_context(
    needle,
    file_content_list: List[Tuple[str, str]],
    position_ratio: float,
    code_context_size: int,
    language: str,
    clean_comments: CleanComment = CleanComment.NoClean,
    context_type: str = "body",
    repo_name: str = "unknown",
    tokenizer = None,
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
    If context_type is "mixed", uses dynamic allocation (complete small + signature large functions).
    If context_type is "optimal", uses fair token distribution with partial function support.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

    # Keep original file content list for needle extraction
    original_file_content_list = file_content_list

    needle_file_idx, needle_file_content = [
        (i, content)
        for i, (f, content) in enumerate(original_file_content_list)
        if f == needle["path"]
    ][0]

    # Handle mixed context type - dynamic allocation (complete small + signature large)
    if context_type == "mixed":
        return _create_mixed_context(
            needle, 
            file_content_list, 
            position_ratio, 
            code_context_size, 
            language, 
            tokenizer,
            repo_name
        )
    
    # Handle optimal context type - fair token distribution with partial functions
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
        # Use our enhanced function extraction that includes class context
        needle_file_functions = extract_functions_from_content(language, needle_file_content)
        
        # Find the needle function and get its enhanced signature
        needle_function = None
        for func in needle_file_functions:
            if func['start_byte'] <= needle["start_byte"] < func['end_byte']:
                needle_function = func
                break
        
        if needle_function:
            needle_code = needle_function['signature']
        else:
            # Fallback to original needle code if we can't find the function
            needle_code = needle_file_content[needle["start_byte"] : needle["end_byte"]]
        
        # Convert all file contents to enhanced function signatures with optimized class grouping
        processed_file_content_list = []
        for path, content in original_file_content_list:
            functions = extract_functions_from_content(language, content)
            
            # Group functions by class to avoid duplicate class definitions
            signatures_parts = []
            current_class = None
            
            for func in functions:
                if func['is_method']:
                    # This is a method - check if we need to add class definition
                    if func['class_name'] != current_class:
                        # New class - add class definition
                        class_def = func['signature'].split('\n')[0]  # First line is class definition
                        signatures_parts.append(class_def)
                        current_class = func['class_name']
                    
                    # Add the method signature (already indented)
                    method_signature = '\n'.join(func['signature'].split('\n')[1:])  # Skip class definition line
                    signatures_parts.append(method_signature)
                else:
                    # Standalone function - add as-is and reset class context
                    signatures_parts.append(func['signature'])
                    current_class = None
            
            signatures = '\n'.join(signatures_parts)
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
        elif context_type in ["optimal", "mixed"]:
            # For optimal and mixed, we don't warn here - warnings are handled in their respective functions
            content_to_tokenize = ""
        else:
            content_to_tokenize = content
        
        total_tokens += len(tokenizer.tokenize(content_to_tokenize))
    
    # Warn if total exceeds context size (skip for optimal and mixed as they handle their own warnings)
    if total_tokens > code_context_size and context_type not in ["optimal", "mixed"]:
        needle_info = f"Function: {needle.get('name', 'unknown')} in {needle.get('path', 'unknown file')}"
        task_info = f"Repo: {repo_name}"
        print(f"⚠️  Warning: {needle_info} ({task_info})")
        print(f"   All files contain {total_tokens} tokens, exceeding context size of {code_context_size}.")
        print(f"   Some files may be truncated in the final context.")
        if context_type == "body":
            print(f"   Consider using --context_type signature, mixed, or optimal to reduce token usage.")

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
        # Add blank line before this file if there's already content
        if code_prefix.strip():
            code_prefix = prefix + "\n" + code_prefix
        else:
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
        # Add blank line before this file if there's already content
        if code_suffix.strip():
            code_suffix += "\n" + suffix
        else:
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
    # Model hyperparameters
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,  # -1 means disabled
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
):
    # Parameter validation
    if not (0.0 <= temperature <= 2.0):
        raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
    if not (0.0 <= top_p <= 1.0):
        raise ValueError(f"top_p must be between 0.0 and 1.0, got {top_p}")
    if not (top_k == -1 or top_k > 0):
        raise ValueError(f"top_k must be -1 (disabled) or positive integer, got {top_k}")
    if not (repetition_penalty >= 0.0):
        raise ValueError(f"repetition_penalty must be >= 0.0, got {repetition_penalty}")
    if not (-2.0 <= presence_penalty <= 2.0):
        raise ValueError(f"presence_penalty must be between -2.0 and 2.0, got {presence_penalty}")
    if not (-2.0 <= frequency_penalty <= 2.0):
        raise ValueError(f"frequency_penalty must be between -2.0 and 2.0, got {frequency_penalty}")

    if backend is None:
        if base_url is not None:
            backend = "openai"
        else:
            backend = "vllm"
        print(f"Using {backend} as the backend")
    assert backend is not None, "Please specify the backend"

    # Provider-specific parameter warnings
    if backend == "openai":
        if top_k != -1:
            print(f"⚠️  Warning: OpenAI does not support top_k parameter. Ignoring top_k={top_k}")
        if repetition_penalty != 1.0:
            print(f"⚠️  Warning: OpenAI does not support repetition_penalty. Use frequency_penalty instead. Ignoring repetition_penalty={repetition_penalty}")
    elif backend == "anthropic":
        if presence_penalty != 0.0:
            print(f"⚠️  Warning: Anthropic does not support presence_penalty parameter. Ignoring presence_penalty={presence_penalty}")
        if frequency_penalty != 0.0:
            print(f"⚠️  Warning: Anthropic does not support frequency_penalty parameter. Ignoring frequency_penalty={frequency_penalty}")
        if repetition_penalty != 1.0:
            print(f"⚠️  Warning: Anthropic does not support repetition_penalty parameter. Ignoring repetition_penalty={repetition_penalty}")
        if temperature > 1.0:
            print(f"⚠️  Warning: Anthropic recommends temperature <= 1.0, got {temperature}")
    elif backend == "google":
        if presence_penalty != 0.0:
            print(f"⚠️  Warning: Google does not support presence_penalty parameter. Ignoring presence_penalty={presence_penalty}")
        if frequency_penalty != 0.0:
            print(f"⚠️  Warning: Google does not support frequency_penalty parameter. Ignoring frequency_penalty={frequency_penalty}")
        if repetition_penalty != 1.0:
            print(f"⚠️  Warning: Google does not support repetition_penalty parameter. Ignoring repetition_penalty={repetition_penalty}")
    # VLLM and HF support all parameters
    
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
    os.makedirs(os.path.join(result_dir, task_type, context_type), exist_ok=True)
    context_size_dir = os.path.join(result_dir, task_type, context_type, f"ntoken_{code_context_size}")
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
        CACHE_DIR, task_type, context_type, f"cache{extra}_ntoken_{code_context_size}_v1.jsonl"
    )
    os.makedirs(os.path.join(CACHE_DIR, task_type, context_type), exist_ok=True)

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
            for repo in pbar.track(repos):
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
                
                # Initialize tokenizer for token calculations (used for prompt overhead and debugging)
                tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
                
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
                    
                    # Calculate prompt overhead tokens and adjust code context size
                    prompt_overhead_tokens = calculate_prompt_overhead_tokens(
                        task_type=task_type,
                        needle_name=needle["name"], 
                        needle_description=task["description"],
                        tokenizer=tokenizer
                    )
                    
                    # Ensure we have enough tokens for code context
                    adjusted_code_context_size = max(1024, code_context_size - prompt_overhead_tokens)
                    
                    
                    # Warn if prompt overhead is very large
                    if prompt_overhead_tokens > code_context_size * 0.1:  # More than 10% overhead
                        print(f"⚠️  Warning: Prompt overhead ({prompt_overhead_tokens} tokens) is {prompt_overhead_tokens/code_context_size*100:.1f}% of context window")
                    
                    code_context_info = make_code_context(
                        needle,
                        file_content_list,
                        position_ratio=position_ratio,
                        code_context_size=adjusted_code_context_size,
                        language=lang,
                        clean_comments=clean_ctx_comments,
                        context_type=context_type,
                        repo_name=repo["repo"],
                        tokenizer=tokenizer,
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
                    prompt, 
                    n=1, 
                    max_tokens=max_new_tokens, 
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    system_msg=system_message
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
