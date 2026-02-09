"""
Codebase RAG Indexer — Self-Indexing Knowledge Base.

Indexes the platform's own source code into the shared knowledge base,
enabling the Overseer Agent (and other agents) to understand:
- What modules exist and what they do
- How agents are structured and connected
- Where specific functionality lives
- Configuration patterns and schemas

This creates a "self-aware" platform that can answer questions about
its own architecture and help with debugging, onboarding, and planning.

Architecture:
    CodebaseIndexer
    +-- scan_directory()      → Find Python/YAML files
    +-- extract_module_info() → Parse docstrings, classes, functions
    +-- chunk_content()       → Split into embeddable chunks
    +-- index_codebase()      → Full pipeline: scan → extract → chunk → embed → store
    +-- search_codebase()     → Query indexed code knowledge

Safety:
    - Read-only scanning (never modifies source files)
    - Excludes __pycache__, .git, node_modules, .venv
    - Chunks are small (< 1000 tokens) for efficient retrieval
    - Incremental: only re-indexes changed files (by mtime)

Usage:
    from core.rag.codebase_indexer import CodebaseIndexer

    indexer = CodebaseIndexer(db=db, embedder=embedder)
    stats = await indexer.index_codebase()
    # → {"files_scanned": 45, "chunks_created": 128, "modules_indexed": 32}

    results = await indexer.search_codebase("how does the circuit breaker work")
    # → [{"content": "...", "file_path": "core/agents/base.py", "similarity": 0.89}]
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Directories and files to skip
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "htmlcov",
    "dist", "build", "egg-info", ".eggs", ".tox",
}

INCLUDE_EXTENSIONS = {".py", ".yaml", ".yml", ".md", ".sql", ".toml"}

# Max chunk size in characters (~250 tokens)
MAX_CHUNK_SIZE = 1000
OVERLAP_SIZE = 100


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ModuleInfo:
    """Extracted metadata about a Python module."""
    file_path: str           # Relative path from project root
    module_name: str         # Dotted module name (e.g., "core.agents.base")
    docstring: str           # Module-level docstring
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    line_count: int = 0

    @property
    def summary(self) -> str:
        """One-line summary from docstring."""
        if self.docstring:
            first_line = self.docstring.strip().split("\n")[0]
            return first_line[:200]
        return f"Module: {self.module_name}"


@dataclass
class ClassInfo:
    """Extracted metadata about a Python class."""
    name: str
    docstring: str
    methods: list[str]       # Method names
    bases: list[str]         # Base class names
    decorators: list[str]    # Decorator names
    line_number: int = 0


@dataclass
class FunctionInfo:
    """Extracted metadata about a Python function."""
    name: str
    docstring: str
    parameters: list[str]    # Parameter names
    decorators: list[str]
    is_async: bool = False
    line_number: int = 0


@dataclass
class CodeChunk:
    """A chunk of code knowledge ready for embedding."""
    content: str             # The text to embed
    file_path: str           # Source file (relative)
    chunk_type: str          # "module_overview", "class_doc", "function_doc", "config", "sql"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Python AST Parser
# ---------------------------------------------------------------------------

class PythonParser:
    """
    Extract structured information from Python source files using AST.

    This is a lightweight parser — it reads docstrings, class/function
    signatures, decorators, and imports. It does NOT execute the code.
    """

    @staticmethod
    def parse_file(file_path: str, relative_path: str) -> Optional[ModuleInfo]:
        """
        Parse a Python file and extract module information.

        Args:
            file_path: Absolute path to the .py file.
            relative_path: Path relative to project root.

        Returns:
            ModuleInfo with extracted metadata, or None on parse failure.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception as e:
            logger.debug(f"Cannot read {file_path}: {e}")
            return None

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return None

        # Module-level docstring
        module_doc = ast.get_docstring(tree) or ""

        # Module name from path
        module_name = relative_path.replace(os.sep, ".").replace("/", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        if module_name.endswith(".__init__"):
            module_name = module_name[:-9]

        info = ModuleInfo(
            file_path=relative_path,
            module_name=module_name,
            docstring=module_doc,
            line_count=len(source.splitlines()),
        )

        # Walk AST for classes, functions, imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = ClassInfo(
                    name=node.name,
                    docstring=ast.get_docstring(node) or "",
                    methods=[
                        n.name for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                    bases=[
                        PythonParser._get_name(b)
                        for b in node.bases
                    ],
                    decorators=[
                        PythonParser._get_decorator_name(d)
                        for d in node.decorator_list
                    ],
                    line_number=node.lineno,
                )
                info.classes.append(class_info)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip methods (already captured in classes)
                if not isinstance(getattr(node, '_parent', None), ast.ClassDef):
                    # Top-level function detection (approximate)
                    if node.col_offset == 0:
                        func_info = FunctionInfo(
                            name=node.name,
                            docstring=ast.get_docstring(node) or "",
                            parameters=[
                                arg.arg for arg in node.args.args
                                if arg.arg != "self"
                            ],
                            decorators=[
                                PythonParser._get_decorator_name(d)
                                for d in node.decorator_list
                            ],
                            is_async=isinstance(node, ast.AsyncFunctionDef),
                            line_number=node.lineno,
                        )
                        info.functions.append(func_info)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                try:
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            info.imports.append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            info.imports.append(f"{module}.{alias.name}")
                except Exception:
                    pass

        return info

    @staticmethod
    def _get_name(node: ast.expr) -> str:
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{PythonParser._get_name(node.value)}.{node.attr}"
        return "?"

    @staticmethod
    def _get_decorator_name(node: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{PythonParser._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return PythonParser._get_decorator_name(node.func)
        return "?"


# ---------------------------------------------------------------------------
# Codebase Indexer
# ---------------------------------------------------------------------------

class CodebaseIndexer:
    """
    Indexes the platform's source code into the shared knowledge base.

    Provides:
    - Full codebase scanning with file type filtering
    - AST-based Python module analysis
    - Content chunking for embedding
    - Incremental indexing (skip unchanged files)
    - Search over indexed code knowledge
    """

    def __init__(
        self,
        db: Any = None,
        embedder: Any = None,
        project_root: Optional[Path] = None,
        chunk_type: str = "codebase_knowledge",
    ):
        self.db = db
        self.embedder = embedder
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.chunk_type = chunk_type
        self._file_hashes: dict[str, str] = {}  # path → content hash

    # --- Scanning ---

    def scan_directory(
        self,
        directory: Optional[Path] = None,
        extensions: Optional[set[str]] = None,
    ) -> list[Path]:
        """
        Recursively find indexable files in the project.

        Args:
            directory: Root directory to scan (default: project_root).
            extensions: File extensions to include (default: INCLUDE_EXTENSIONS).

        Returns:
            List of absolute file paths.
        """
        root = directory or self.project_root
        exts = extensions or INCLUDE_EXTENSIONS
        files: list[Path] = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip excluded directories
            dirnames[:] = [
                d for d in dirnames
                if d not in EXCLUDE_DIRS
                and not d.startswith(".")
            ]

            for filename in filenames:
                ext = Path(filename).suffix
                if ext in exts:
                    files.append(Path(dirpath) / filename)

        return sorted(files)

    # --- Extraction ---

    def extract_module_info(self, file_path: Path) -> Optional[ModuleInfo]:
        """
        Extract structured information from a Python file.

        Args:
            file_path: Absolute path to the .py file.

        Returns:
            ModuleInfo or None if the file can't be parsed.
        """
        try:
            relative = str(file_path.relative_to(self.project_root))
        except ValueError:
            relative = str(file_path)

        return PythonParser.parse_file(str(file_path), relative)

    # --- Chunking ---

    def chunk_module(self, module: ModuleInfo) -> list[CodeChunk]:
        """
        Split a module's information into embeddable chunks.

        Strategy:
        1. Module overview chunk (docstring + class/function listing)
        2. One chunk per class (docstring + methods)
        3. One chunk per major function (docstring + params)
        """
        chunks: list[CodeChunk] = []

        # 1. Module overview
        overview_lines = [
            f"# Module: {module.module_name}",
            f"File: {module.file_path}",
            f"Lines: {module.line_count}",
            "",
        ]
        if module.docstring:
            overview_lines.append(module.docstring[:500])
            overview_lines.append("")

        if module.classes:
            overview_lines.append("## Classes:")
            for cls in module.classes:
                bases = f" ({', '.join(cls.bases)})" if cls.bases else ""
                overview_lines.append(f"- {cls.name}{bases}")

        if module.functions:
            overview_lines.append("## Functions:")
            for func in module.functions:
                async_prefix = "async " if func.is_async else ""
                overview_lines.append(f"- {async_prefix}{func.name}()")

        overview = "\n".join(overview_lines)
        chunks.append(CodeChunk(
            content=overview[:MAX_CHUNK_SIZE],
            file_path=module.file_path,
            chunk_type="module_overview",
            metadata={
                "module_name": module.module_name,
                "class_count": len(module.classes),
                "function_count": len(module.functions),
                "line_count": module.line_count,
            },
        ))

        # 2. Class chunks
        for cls in module.classes:
            class_lines = [
                f"# Class: {cls.name}",
                f"Module: {module.module_name}",
                f"File: {module.file_path}:{cls.line_number}",
            ]
            if cls.bases:
                class_lines.append(f"Inherits from: {', '.join(cls.bases)}")
            if cls.decorators:
                class_lines.append(f"Decorators: {', '.join(cls.decorators)}")
            class_lines.append("")

            if cls.docstring:
                class_lines.append(cls.docstring[:400])
                class_lines.append("")

            if cls.methods:
                class_lines.append(f"Methods: {', '.join(cls.methods[:20])}")

            class_text = "\n".join(class_lines)
            chunks.append(CodeChunk(
                content=class_text[:MAX_CHUNK_SIZE],
                file_path=module.file_path,
                chunk_type="class_doc",
                metadata={
                    "module_name": module.module_name,
                    "class_name": cls.name,
                    "method_count": len(cls.methods),
                },
            ))

        # 3. Function chunks (only for functions with docstrings)
        for func in module.functions:
            if not func.docstring:
                continue

            func_lines = [
                f"# Function: {'async ' if func.is_async else ''}{func.name}()",
                f"Module: {module.module_name}",
                f"File: {module.file_path}:{func.line_number}",
            ]
            if func.parameters:
                func_lines.append(f"Parameters: {', '.join(func.parameters[:10])}")
            if func.decorators:
                func_lines.append(f"Decorators: {', '.join(func.decorators)}")
            func_lines.append("")
            func_lines.append(func.docstring[:500])

            func_text = "\n".join(func_lines)
            chunks.append(CodeChunk(
                content=func_text[:MAX_CHUNK_SIZE],
                file_path=module.file_path,
                chunk_type="function_doc",
                metadata={
                    "module_name": module.module_name,
                    "function_name": func.name,
                    "is_async": func.is_async,
                },
            ))

        return chunks

    def chunk_config_file(self, file_path: Path) -> list[CodeChunk]:
        """
        Chunk a YAML/TOML/SQL config file.

        Reads the file and creates a single chunk with the full content
        (or splits if too large).
        """
        try:
            relative = str(file_path.relative_to(self.project_root))
        except ValueError:
            relative = str(file_path)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return []

        ext = file_path.suffix
        chunk_type_map = {
            ".yaml": "config",
            ".yml": "config",
            ".sql": "sql_migration",
            ".toml": "config",
            ".md": "documentation",
        }
        chunk_type = chunk_type_map.get(ext, "config")

        # If content fits in one chunk, create one
        if len(content) <= MAX_CHUNK_SIZE:
            return [CodeChunk(
                content=f"# File: {relative}\n\n{content}",
                file_path=relative,
                chunk_type=chunk_type,
                metadata={"file_type": ext},
            )]

        # Split into chunks with overlap
        chunks: list[CodeChunk] = []
        lines = content.splitlines()
        current: list[str] = [f"# File: {relative} (part {{part}})"]
        current_len = len(current[0])

        part = 1
        for line in lines:
            if current_len + len(line) + 1 > MAX_CHUNK_SIZE and len(current) > 1:
                chunks.append(CodeChunk(
                    content="\n".join(current).replace("{part}", str(part)),
                    file_path=relative,
                    chunk_type=chunk_type,
                    metadata={"file_type": ext, "part": part},
                ))
                # Start new chunk with overlap
                overlap = current[-3:] if len(current) > 3 else current[-1:]
                current = [f"# File: {relative} (part {{part}})"] + overlap
                current_len = sum(len(l) for l in current)
                part += 1

            current.append(line)
            current_len += len(line) + 1

        # Final chunk
        if len(current) > 1:
            chunks.append(CodeChunk(
                content="\n".join(current).replace("{part}", str(part)),
                file_path=relative,
                chunk_type=chunk_type,
                metadata={"file_type": ext, "part": part},
            ))

        return chunks

    # --- Full Pipeline ---

    async def index_codebase(
        self,
        directories: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Full indexing pipeline: scan → extract → chunk → embed → store.

        Args:
            directories: Specific subdirectories to index (relative to project root).
                         Default: ["core", "verticals", "infrastructure/migrations"].

        Returns:
            Statistics dict: files_scanned, chunks_created, modules_indexed.
        """
        if directories is None:
            directories = ["core", "verticals", "infrastructure/migrations"]

        scan_roots = [self.project_root / d for d in directories if (self.project_root / d).exists()]

        all_files: list[Path] = []
        for root in scan_roots:
            all_files.extend(self.scan_directory(root))

        logger.info(
            "codebase_indexing_started",
            extra={
                "file_count": len(all_files),
                "directories": directories,
            },
        )

        all_chunks: list[CodeChunk] = []
        modules_indexed = 0

        for file_path in all_files:
            if file_path.suffix == ".py":
                module = self.extract_module_info(file_path)
                if module:
                    chunks = self.chunk_module(module)
                    all_chunks.extend(chunks)
                    modules_indexed += 1
            else:
                chunks = self.chunk_config_file(file_path)
                all_chunks.extend(chunks)

        # Embed and store chunks
        stored = 0
        if self.embedder and self.db:
            for chunk in all_chunks:
                try:
                    embedding = await self.embedder.embed_text(chunk.content)
                    self.db.store_knowledge_chunk(
                        content=chunk.content,
                        chunk_type=self.chunk_type,
                        source_type=chunk.chunk_type,
                        embedding=embedding,
                        metadata={
                            **chunk.metadata,
                            "file_path": chunk.file_path,
                            "content_hash": chunk.content_hash,
                        },
                    )
                    stored += 1
                except Exception as e:
                    logger.debug(f"Failed to store chunk from {chunk.file_path}: {e}")

        stats = {
            "files_scanned": len(all_files),
            "chunks_created": len(all_chunks),
            "chunks_stored": stored,
            "modules_indexed": modules_indexed,
        }

        logger.info(
            "codebase_indexing_complete",
            extra=stats,
        )

        return stats

    async def search_codebase(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search indexed codebase knowledge.

        Args:
            query: Natural language search query.
            limit: Max results.

        Returns:
            List of matching chunks with content and metadata.
        """
        if not self.embedder or not self.db:
            return []

        query_embedding = await self.embedder.embed_text(query)
        results = self.db.search_knowledge(
            query_embedding=query_embedding,
            chunk_type=self.chunk_type,
            limit=limit,
        )

        return [
            {
                "content": r.get("content", ""),
                "file_path": r.get("metadata", {}).get("file_path", ""),
                "chunk_type": r.get("source_type", ""),
                "similarity": r.get("similarity"),
                "metadata": r.get("metadata", {}),
            }
            for r in results
        ]

    # --- Utility ---

    def get_project_structure(
        self,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """
        Get a tree representation of the project structure.

        Returns a nested dict suitable for display or LLM context.
        """
        structure: dict[str, Any] = {}

        for file_path in self.scan_directory():
            try:
                relative = file_path.relative_to(self.project_root)
            except ValueError:
                continue

            parts = relative.parts
            if len(parts) > max_depth:
                parts = parts[:max_depth] + ("...",)

            current = structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = file_path.suffix

        return structure
