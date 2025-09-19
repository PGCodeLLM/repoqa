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