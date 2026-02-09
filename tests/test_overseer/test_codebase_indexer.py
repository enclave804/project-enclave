"""
Tests for codebase_indexer.py — Self-Indexing Knowledge Base.

Tests:
- PythonParser: AST parsing of Python files
- CodebaseIndexer: file scanning, module extraction, chunking
- ModuleInfo: module overview generation
- ClassInfo & FunctionInfo: extraction from AST
- CodeChunk: content hashing, chunk types
- Directory scanning: include/exclude patterns
- Config file chunking: YAML, SQL, TOML
- Full pipeline: scan → extract → chunk (without DB)
"""

import os
import tempfile
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.rag.codebase_indexer import (
    CodebaseIndexer,
    CodeChunk,
    ClassInfo,
    FunctionInfo,
    ModuleInfo,
    PythonParser,
    EXCLUDE_DIRS,
    INCLUDE_EXTENSIONS,
    MAX_CHUNK_SIZE,
)


# ===========================================================================
# PythonParser Tests
# ===========================================================================

class TestPythonParser:
    """Tests for AST-based Python file parsing."""

    def _write_temp_py(self, content: str) -> tuple[str, str]:
        """Write content to a temp .py file, return (abs_path, rel_path)."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        )
        tmp.write(dedent(content))
        tmp.close()
        return tmp.name, os.path.basename(tmp.name)

    def test_parse_module_docstring(self):
        abs_path, rel_path = self._write_temp_py('''
            """Module-level docstring."""
            x = 1
        ''')
        try:
            info = PythonParser.parse_file(abs_path, rel_path)
            assert info is not None
            assert info.docstring == "Module-level docstring."
        finally:
            os.unlink(abs_path)

    def test_parse_class(self):
        abs_path, rel_path = self._write_temp_py('''
            """Test module."""

            class MyAgent:
                """An agent that does things."""
                def run(self):
                    pass
                async def process(self):
                    pass
        ''')
        try:
            info = PythonParser.parse_file(abs_path, rel_path)
            assert len(info.classes) == 1
            cls = info.classes[0]
            assert cls.name == "MyAgent"
            assert cls.docstring == "An agent that does things."
            assert "run" in cls.methods
            assert "process" in cls.methods
        finally:
            os.unlink(abs_path)

    def test_parse_class_with_bases(self):
        abs_path, rel_path = self._write_temp_py('''
            """Test."""
            class Child(BaseAgent):
                """Child class."""
                pass
        ''')
        try:
            info = PythonParser.parse_file(abs_path, rel_path)
            assert len(info.classes) == 1
            assert "BaseAgent" in info.classes[0].bases
        finally:
            os.unlink(abs_path)

    def test_parse_class_with_decorators(self):
        abs_path, rel_path = self._write_temp_py('''
            """Test."""
            def register(name):
                def dec(cls):
                    return cls
                return dec

            @register("test")
            class MyClass:
                """Decorated."""
                pass
        ''')
        try:
            info = PythonParser.parse_file(abs_path, rel_path)
            assert len(info.classes) == 1
            assert info.classes[0].decorators  # Has at least one decorator
        finally:
            os.unlink(abs_path)

    def test_parse_top_level_function(self):
        abs_path, rel_path = self._write_temp_py('''
            """Module doc."""

            def helper_function(x, y):
                """Add two numbers."""
                return x + y

            async def async_helper(query):
                """Search async."""
                pass
        ''')
        try:
            info = PythonParser.parse_file(abs_path, rel_path)
            # Should find top-level functions
            func_names = [f.name for f in info.functions]
            assert "helper_function" in func_names or "async_helper" in func_names
        finally:
            os.unlink(abs_path)

    def test_parse_imports(self):
        abs_path, rel_path = self._write_temp_py('''
            """Test."""
            import os
            from pathlib import Path
            from core.agents.base import BaseAgent
        ''')
        try:
            info = PythonParser.parse_file(abs_path, rel_path)
            assert "os" in info.imports
            assert any("Path" in imp for imp in info.imports)
            assert any("BaseAgent" in imp for imp in info.imports)
        finally:
            os.unlink(abs_path)

    def test_parse_syntax_error(self):
        abs_path, rel_path = self._write_temp_py('''
            def broken(:
                pass
        ''')
        try:
            info = PythonParser.parse_file(abs_path, rel_path)
            assert info is None  # Should return None on syntax error
        finally:
            os.unlink(abs_path)

    def test_parse_nonexistent_file(self):
        info = PythonParser.parse_file("/nonexistent/path.py", "path.py")
        assert info is None

    def test_line_count(self):
        abs_path, rel_path = self._write_temp_py('''
            """Test."""
            x = 1
            y = 2
            z = 3
        ''')
        try:
            info = PythonParser.parse_file(abs_path, rel_path)
            assert info.line_count > 0
        finally:
            os.unlink(abs_path)

    def test_module_name_from_path(self):
        abs_path, rel_path = self._write_temp_py('"""Test."""\n')
        try:
            info = PythonParser.parse_file(abs_path, "core/agents/base.py")
            assert info.module_name == "core.agents.base"
        finally:
            os.unlink(abs_path)


# ===========================================================================
# CodeChunk Tests
# ===========================================================================

class TestCodeChunk:
    """Tests for code chunks."""

    def test_content_hash_deterministic(self):
        chunk1 = CodeChunk(content="hello", file_path="test.py", chunk_type="module_overview")
        chunk2 = CodeChunk(content="hello", file_path="test.py", chunk_type="module_overview")
        assert chunk1.content_hash == chunk2.content_hash

    def test_different_content_different_hash(self):
        chunk1 = CodeChunk(content="hello", file_path="test.py", chunk_type="module_overview")
        chunk2 = CodeChunk(content="world", file_path="test.py", chunk_type="module_overview")
        assert chunk1.content_hash != chunk2.content_hash

    def test_metadata_default(self):
        chunk = CodeChunk(content="test", file_path="f.py", chunk_type="class_doc")
        assert chunk.metadata == {}


# ===========================================================================
# ModuleInfo Tests
# ===========================================================================

class TestModuleInfo:
    """Tests for ModuleInfo data model."""

    def test_summary_from_docstring(self):
        info = ModuleInfo(
            file_path="core/agents/base.py",
            module_name="core.agents.base",
            docstring="Base agent class for the platform.\n\nMore details here.",
        )
        assert info.summary == "Base agent class for the platform."

    def test_summary_without_docstring(self):
        info = ModuleInfo(
            file_path="core/agents/base.py",
            module_name="core.agents.base",
            docstring="",
        )
        assert "core.agents.base" in info.summary


# ===========================================================================
# CodebaseIndexer Tests
# ===========================================================================

class TestCodebaseIndexer:
    """Tests for the CodebaseIndexer class."""

    def _create_test_project(self) -> Path:
        """Create a temporary project structure for testing."""
        root = Path(tempfile.mkdtemp())

        # Create Python files
        (root / "core").mkdir()
        (root / "core" / "__init__.py").write_text('"""Core package."""\n')
        (root / "core" / "agent.py").write_text(dedent('''\
            """Agent module with classes."""

            class BaseAgent:
                """Base class for all agents."""
                def run(self):
                    """Execute the agent."""
                    pass
                def get_tools(self):
                    """Return tools."""
                    return []

            class SpecialAgent(BaseAgent):
                """A specialized agent."""
                async def process(self):
                    """Process something."""
                    pass
        '''))

        # Create config file
        (root / "config.yaml").write_text("name: test\nversion: 1.0\n")

        # Create SQL migration
        (root / "migrations").mkdir()
        (root / "migrations" / "001.sql").write_text(
            "CREATE TABLE agents (id TEXT PRIMARY KEY);\n"
        )

        # Create excluded directory
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "test.pyc").write_text("bytecode")

        return root

    def test_scan_directory(self):
        root = self._create_test_project()
        try:
            indexer = CodebaseIndexer(project_root=root)
            files = indexer.scan_directory()

            extensions = {f.suffix for f in files}
            assert ".py" in extensions
            assert ".yaml" in extensions
            assert ".sql" in extensions

            # __pycache__ should be excluded
            paths = [str(f) for f in files]
            assert not any("__pycache__" in p for p in paths)
        finally:
            import shutil
            shutil.rmtree(root)

    def test_extract_module_info(self):
        root = self._create_test_project()
        try:
            indexer = CodebaseIndexer(project_root=root)
            info = indexer.extract_module_info(root / "core" / "agent.py")

            assert info is not None
            assert info.module_name == "core.agent"
            assert "Agent module" in info.docstring
            assert len(info.classes) == 2

            base = next(c for c in info.classes if c.name == "BaseAgent")
            assert "run" in base.methods
            assert "get_tools" in base.methods

            special = next(c for c in info.classes if c.name == "SpecialAgent")
            assert "BaseAgent" in special.bases
        finally:
            import shutil
            shutil.rmtree(root)

    def test_chunk_module(self):
        root = self._create_test_project()
        try:
            indexer = CodebaseIndexer(project_root=root)
            info = indexer.extract_module_info(root / "core" / "agent.py")
            chunks = indexer.chunk_module(info)

            # Should have: 1 module overview + 2 class chunks
            assert len(chunks) >= 3

            # Check chunk types
            types = {c.chunk_type for c in chunks}
            assert "module_overview" in types
            assert "class_doc" in types

            # Module overview should list classes
            overview = next(c for c in chunks if c.chunk_type == "module_overview")
            assert "BaseAgent" in overview.content
            assert "SpecialAgent" in overview.content

            # Class chunk should have methods
            base_chunk = next(
                c for c in chunks
                if c.chunk_type == "class_doc" and "BaseAgent" in c.content
            )
            assert "run" in base_chunk.content
        finally:
            import shutil
            shutil.rmtree(root)

    def test_chunk_config_file(self):
        root = self._create_test_project()
        try:
            indexer = CodebaseIndexer(project_root=root)
            chunks = indexer.chunk_config_file(root / "config.yaml")

            assert len(chunks) == 1
            assert chunks[0].chunk_type == "config"
            assert "name: test" in chunks[0].content
        finally:
            import shutil
            shutil.rmtree(root)

    def test_chunk_sql_file(self):
        root = self._create_test_project()
        try:
            indexer = CodebaseIndexer(project_root=root)
            chunks = indexer.chunk_config_file(root / "migrations" / "001.sql")

            assert len(chunks) == 1
            assert chunks[0].chunk_type == "sql_migration"
            assert "CREATE TABLE" in chunks[0].content
        finally:
            import shutil
            shutil.rmtree(root)

    def test_large_file_chunking(self):
        """Large files should be split into multiple chunks."""
        root = Path(tempfile.mkdtemp())
        try:
            large_content = "-- SQL line\n" * 200  # ~2400 chars
            (root / "large.sql").write_text(large_content)

            indexer = CodebaseIndexer(project_root=root)
            chunks = indexer.chunk_config_file(root / "large.sql")

            assert len(chunks) > 1
            for chunk in chunks:
                assert len(chunk.content) <= MAX_CHUNK_SIZE + 200  # Allow header
        finally:
            import shutil
            shutil.rmtree(root)

    def test_get_project_structure(self):
        root = self._create_test_project()
        try:
            indexer = CodebaseIndexer(project_root=root)
            structure = indexer.get_project_structure()

            assert "core" in structure
            assert isinstance(structure["core"], dict)
        finally:
            import shutil
            shutil.rmtree(root)


class TestCodebaseIndexerPipeline:
    """Tests for the full indexing pipeline."""

    @pytest.mark.asyncio
    async def test_index_without_db(self):
        """Index pipeline works without DB (creates chunks but doesn't store)."""
        root = Path(tempfile.mkdtemp())
        try:
            (root / "core").mkdir()
            (root / "core" / "test.py").write_text(dedent('''\
                """Test module."""
                class TestClass:
                    """A test class."""
                    pass
            '''))

            indexer = CodebaseIndexer(project_root=root)
            stats = await indexer.index_codebase(directories=["core"])

            assert stats["files_scanned"] == 1
            assert stats["chunks_created"] >= 1
            assert stats["modules_indexed"] == 1
            assert stats["chunks_stored"] == 0  # No DB
        finally:
            import shutil
            shutil.rmtree(root)

    @pytest.mark.asyncio
    async def test_index_with_mock_db(self):
        """Full pipeline with mock DB and embedder."""
        root = Path(tempfile.mkdtemp())
        try:
            (root / "core").mkdir()
            (root / "core" / "agent.py").write_text(dedent('''\
                """Agent module."""
                class Agent:
                    """The agent."""
                    def run(self):
                        """Run it."""
                        pass
            '''))

            mock_db = MagicMock()
            mock_db.store_knowledge_chunk.return_value = {"id": "chunk_1"}

            mock_embedder = AsyncMock()
            mock_embedder.embed_text.return_value = [0.1] * 1536

            indexer = CodebaseIndexer(
                db=mock_db,
                embedder=mock_embedder,
                project_root=root,
            )
            stats = await indexer.index_codebase(directories=["core"])

            assert stats["files_scanned"] == 1
            assert stats["chunks_created"] >= 1
            assert stats["chunks_stored"] >= 1
            mock_db.store_knowledge_chunk.assert_called()
            mock_embedder.embed_text.assert_called()
        finally:
            import shutil
            shutil.rmtree(root)

    @pytest.mark.asyncio
    async def test_search_codebase(self):
        """Search returns formatted results."""
        mock_db = MagicMock()
        mock_db.search_knowledge.return_value = [
            {
                "content": "# Module: core.agents.base\nBase agent class...",
                "source_type": "module_overview",
                "similarity": 0.92,
                "metadata": {"file_path": "core/agents/base.py"},
            },
        ]

        mock_embedder = AsyncMock()
        mock_embedder.embed_text.return_value = [0.1] * 1536

        indexer = CodebaseIndexer(db=mock_db, embedder=mock_embedder)
        results = await indexer.search_codebase("how does the base agent work")

        assert len(results) == 1
        assert results[0]["file_path"] == "core/agents/base.py"
        assert results[0]["similarity"] == 0.92

    @pytest.mark.asyncio
    async def test_search_without_db(self):
        """Search returns empty list when no DB configured."""
        indexer = CodebaseIndexer()
        results = await indexer.search_codebase("anything")
        assert results == []


# ===========================================================================
# Real Project Tests (against actual codebase)
# ===========================================================================

class TestRealCodebase:
    """Tests that scan the real Sovereign Venture Engine codebase."""

    def test_scans_core_directory(self):
        """Verify we can scan the actual core/ directory."""
        project_root = Path(__file__).parent.parent.parent
        core_dir = project_root / "core"

        if not core_dir.exists():
            pytest.skip("core/ directory not found")

        indexer = CodebaseIndexer(project_root=project_root)
        files = indexer.scan_directory(core_dir)

        assert len(files) > 10, f"Expected >10 Python files, found {len(files)}"

        # Verify we found key files
        file_names = {f.name for f in files}
        assert "base.py" in file_names  # core/agents/base.py

    def test_parses_real_base_agent(self):
        """Verify we can parse the real BaseAgent file."""
        project_root = Path(__file__).parent.parent.parent
        base_py = project_root / "core" / "agents" / "base.py"

        if not base_py.exists():
            pytest.skip("core/agents/base.py not found")

        indexer = CodebaseIndexer(project_root=project_root)
        info = indexer.extract_module_info(base_py)

        assert info is not None
        assert info.module_name == "core.agents.base"

        # Should find BaseAgent class
        class_names = [c.name for c in info.classes]
        assert "BaseAgent" in class_names

        # BaseAgent should have key methods
        base_cls = next(c for c in info.classes if c.name == "BaseAgent")
        assert "run" in base_cls.methods or "build_graph" in base_cls.methods

    def test_chunks_real_module(self):
        """Verify chunking produces reasonable output for a real file."""
        project_root = Path(__file__).parent.parent.parent
        base_py = project_root / "core" / "agents" / "base.py"

        if not base_py.exists():
            pytest.skip("core/agents/base.py not found")

        indexer = CodebaseIndexer(project_root=project_root)
        info = indexer.extract_module_info(base_py)
        chunks = indexer.chunk_module(info)

        # Should produce at least module overview + BaseAgent class chunk
        assert len(chunks) >= 2

        # All chunks should be within size limit
        for chunk in chunks:
            assert len(chunk.content) <= MAX_CHUNK_SIZE + 50

        # Module overview should exist
        overview = next(c for c in chunks if c.chunk_type == "module_overview")
        assert "BaseAgent" in overview.content
