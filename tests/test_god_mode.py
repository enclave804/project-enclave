"""
Unit tests for God Mode Lite: RLHF Data Collection + Shadow Agent Infrastructure.

Tests cover:
- Shadow mode fields on AgentInstanceConfig
- BaseAgent.learn() method (RLHF data capture)
- BaseAgent.get_training_examples() retrieval
- TaskQueueManager shadow dispatch
- Supabase client training example methods
"""

import pytest
from unittest.mock import MagicMock, call, patch

from core.config.agent_schema import AgentInstanceConfig
from core.agents.task_queue import TaskQueueManager


# ─── AgentInstanceConfig Shadow Mode ─────────────────────────────────

class TestShadowModeConfig:
    """Tests for shadow_mode and shadow_of config fields."""

    def _make_config(self, **overrides):
        defaults = {
            "agent_id": "test_agent",
            "agent_type": "outreach",
            "name": "Test Agent",
        }
        defaults.update(overrides)
        return AgentInstanceConfig(**defaults)

    def test_shadow_mode_defaults_false(self):
        """shadow_mode should default to False."""
        config = self._make_config()
        assert config.shadow_mode is False

    def test_shadow_of_defaults_none(self):
        """shadow_of should default to None."""
        config = self._make_config()
        assert config.shadow_of is None

    def test_shadow_mode_enabled(self):
        """Should accept shadow_mode=True."""
        config = self._make_config(shadow_mode=True, shadow_of="outreach")
        assert config.shadow_mode is True
        assert config.shadow_of == "outreach"

    def test_shadow_of_without_shadow_mode(self):
        """shadow_of can be set even without shadow_mode (config doesn't enforce coupling)."""
        config = self._make_config(shadow_of="outreach")
        assert config.shadow_of == "outreach"
        assert config.shadow_mode is False

    def test_shadow_config_in_yaml_dict(self):
        """Should parse shadow fields from a YAML-like dict."""
        data = {
            "agent_id": "shadow_outreach_v2",
            "agent_type": "outreach",
            "name": "Shadow Outreach V2",
            "shadow_mode": True,
            "shadow_of": "outreach",
            "description": "Testing new email strategy on real data",
        }
        config = AgentInstanceConfig(**data)
        assert config.shadow_mode is True
        assert config.shadow_of == "outreach"
        assert config.agent_id == "shadow_outreach_v2"

    def test_shadow_mode_serialization(self):
        """Shadow fields should appear in model_dump()."""
        config = self._make_config(shadow_mode=True, shadow_of="outreach")
        data = config.model_dump()
        assert data["shadow_mode"] is True
        assert data["shadow_of"] == "outreach"


# ─── BaseAgent RLHF Methods ─────────────────────────────────────────

class TestBaseAgentLearn:
    """Tests for BaseAgent.learn() RLHF data collection."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.store_training_example.return_value = {
            "id": "te-001",
            "agent_id": "outreach",
            "model_output": "Dear Sir...",
            "score": 85,
        }
        db.get_training_examples.return_value = [
            {"id": "te-001", "model_output": "Draft A", "score": 90},
            {"id": "te-002", "model_output": "Draft B", "score": 75},
        ]
        # BaseAgent.__init__ calls these are best-effort, stub them
        db.log_agent_run = MagicMock()
        db.reset_agent_errors = MagicMock()
        db.record_agent_error = MagicMock()
        return db

    @pytest.fixture
    def agent(self, mock_db):
        """Create a minimal concrete agent for testing learn()."""
        from core.agents.base import BaseAgent
        from core.agents.state import BaseAgentState

        class _TestAgent(BaseAgent):
            agent_type = "test"

            def build_graph(self):
                return MagicMock()

            def get_tools(self):
                return []

            def get_state_class(self):
                return BaseAgentState

        config = AgentInstanceConfig(
            agent_id="outreach",
            agent_type="test",
            name="Test Outreach",
            vertical_id="enclave_guard",
        )
        return _TestAgent(
            config=config,
            db=mock_db,
            embedder=MagicMock(),
            anthropic_client=MagicMock(),
        )

    def test_learn_stores_training_example(self, agent, mock_db):
        """learn() should call db.store_training_example with correct args."""
        result = agent.learn(
            task_input={"lead": "jane@acme.com"},
            model_output="Dear Jane, I hope this email finds you well...",
            human_correction="Hi Jane, noticed your team expanded recently...",
            score=85,
            source="manual_review",
            metadata={"industry": "cybersecurity"},
        )

        assert result is not None
        assert result["id"] == "te-001"
        mock_db.store_training_example.assert_called_once_with(
            agent_id="outreach",
            vertical_id="enclave_guard",
            task_input={"lead": "jane@acme.com"},
            model_output="Dear Jane, I hope this email finds you well...",
            human_correction="Hi Jane, noticed your team expanded recently...",
            score=85,
            source="manual_review",
            metadata={"industry": "cybersecurity"},
        )

    def test_learn_with_score_only(self, agent, mock_db):
        """learn() should work without human_correction (score only)."""
        agent.learn(
            task_input={"lead": "bob@corp.com"},
            model_output="Some draft...",
            score=40,
        )
        call_kwargs = mock_db.store_training_example.call_args
        assert call_kwargs.kwargs["human_correction"] is None
        assert call_kwargs.kwargs["score"] == 40

    def test_learn_minimal_args(self, agent, mock_db):
        """learn() should work with just task_input and model_output."""
        agent.learn(
            task_input={"prompt": "Write email"},
            model_output="Hello...",
        )
        call_kwargs = mock_db.store_training_example.call_args
        assert call_kwargs.kwargs["human_correction"] is None
        assert call_kwargs.kwargs["score"] is None
        assert call_kwargs.kwargs["source"] == "manual_review"
        assert call_kwargs.kwargs["metadata"] == {}

    def test_learn_never_crashes_on_db_failure(self, agent, mock_db):
        """learn() should return None on DB error, never raise."""
        mock_db.store_training_example.side_effect = Exception("DB connection lost")
        result = agent.learn(
            task_input={"x": 1},
            model_output="test",
        )
        assert result is None  # Graceful failure

    def test_learn_source_types(self, agent, mock_db):
        """learn() should pass through different source types."""
        for source in ["manual_review", "shadow_comparison", "a_b_test", "automated_eval"]:
            agent.learn(
                task_input={},
                model_output="draft",
                source=source,
            )
        assert mock_db.store_training_example.call_count == 4

    def test_get_training_examples_returns_list(self, agent, mock_db):
        """get_training_examples() should return list of records."""
        examples = agent.get_training_examples(min_score=70, limit=50)
        assert len(examples) == 2
        assert examples[0]["score"] == 90
        mock_db.get_training_examples.assert_called_once_with(
            agent_id="outreach",
            vertical_id="enclave_guard",
            min_score=70,
            source=None,
            limit=50,
        )

    def test_get_training_examples_with_source_filter(self, agent, mock_db):
        """get_training_examples() should pass source filter."""
        agent.get_training_examples(source="shadow_comparison")
        call_kwargs = mock_db.get_training_examples.call_args
        assert call_kwargs.kwargs["source"] == "shadow_comparison"

    def test_get_training_examples_returns_empty_on_failure(self, agent, mock_db):
        """get_training_examples() should return [] on DB error."""
        mock_db.get_training_examples.side_effect = Exception("RPC timeout")
        result = agent.get_training_examples()
        assert result == []


# ─── TaskQueueManager Shadow Dispatch ────────────────────────────────

class TestShadowDispatch:
    """Tests for shadow task dispatch in the task queue."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.enqueue_task.return_value = {
            "task_id": "task-primary",
            "target_agent_id": "outreach",
            "status": "pending",
        }
        db.get_shadow_agents.return_value = []
        return db

    @pytest.fixture
    def queue(self, mock_db):
        return TaskQueueManager(mock_db)

    def test_enqueue_calls_shadow_dispatch(self, queue, mock_db):
        """Enqueue should check for shadow agents."""
        queue.enqueue("outreach", "process_lead", {"lead": "test"})
        mock_db.get_shadow_agents.assert_called_once_with("outreach")

    def test_enqueue_creates_shadow_copies(self, queue, mock_db):
        """Should create duplicate tasks for each shadow agent."""
        mock_db.get_shadow_agents.return_value = [
            {"agent_id": "shadow_outreach_v2", "enabled": True},
            {"agent_id": "shadow_outreach_v3", "enabled": True},
        ]

        queue.enqueue("outreach", "process_lead", {"lead": "jane@acme.com"})

        # Primary task + 2 shadow tasks = 3 enqueue calls
        assert mock_db.enqueue_task.call_count == 3

        # Verify shadow tasks have correct metadata
        shadow_calls = mock_db.enqueue_task.call_args_list[1:]  # Skip primary
        for shadow_call in shadow_calls:
            shadow_data = shadow_call[0][0]
            assert shadow_data["metadata"]["shadow_mode"] is True
            assert shadow_data["metadata"]["champion_agent_id"] == "outreach"
            assert shadow_data["task_type"] == "process_lead"
            assert shadow_data["input_data"] == {"lead": "jane@acme.com"}

    def test_shadow_targets_correct_agent_ids(self, queue, mock_db):
        """Each shadow task should target the correct shadow agent."""
        mock_db.get_shadow_agents.return_value = [
            {"agent_id": "shadow_v2"},
            {"agent_id": "shadow_v3"},
        ]

        queue.enqueue("outreach", "process_lead", {})

        shadow_calls = mock_db.enqueue_task.call_args_list[1:]
        target_ids = [c[0][0]["target_agent_id"] for c in shadow_calls]
        assert "shadow_v2" in target_ids
        assert "shadow_v3" in target_ids

    def test_no_shadow_copies_for_shadow_tasks(self, queue, mock_db):
        """Shadow tasks should NOT create more shadow copies (no recursion)."""
        mock_db.get_shadow_agents.return_value = [
            {"agent_id": "shadow_of_shadow"},
        ]

        queue.enqueue(
            "shadow_v2",
            "process_lead",
            {},
            metadata={"shadow_mode": True, "champion_agent_id": "outreach"},
        )

        # Only the primary enqueue — no shadow dispatch
        assert mock_db.enqueue_task.call_count == 1
        mock_db.get_shadow_agents.assert_not_called()

    def test_shadow_dispatch_handles_no_shadows(self, queue, mock_db):
        """Should work fine when no shadow agents exist."""
        mock_db.get_shadow_agents.return_value = []
        result = queue.enqueue("outreach", "process_lead", {})
        assert result["task_id"] == "task-primary"
        assert mock_db.enqueue_task.call_count == 1  # Primary only

    def test_shadow_dispatch_handles_db_method_missing(self, queue, mock_db):
        """Should gracefully handle missing get_shadow_agents method."""
        mock_db.get_shadow_agents.side_effect = AttributeError(
            "'MockDB' has no attribute 'get_shadow_agents'"
        )
        # Should still return the primary task
        result = queue.enqueue("outreach", "process_lead", {"data": 1})
        assert result["task_id"] == "task-primary"
        assert mock_db.enqueue_task.call_count == 1

    def test_shadow_dispatch_handles_db_error(self, queue, mock_db):
        """Shadow dispatch failure should not affect primary task."""
        mock_db.get_shadow_agents.side_effect = Exception("DB connection error")
        result = queue.enqueue("outreach", "process_lead", {})
        assert result["task_id"] == "task-primary"

    def test_shadow_task_enqueue_failure_continues(self, queue, mock_db):
        """If one shadow task fails to enqueue, others should still be created."""
        mock_db.get_shadow_agents.return_value = [
            {"agent_id": "shadow_a"},
            {"agent_id": "shadow_b"},
        ]
        # Primary succeeds, first shadow fails, second succeeds
        mock_db.enqueue_task.side_effect = [
            {"task_id": "primary"},  # Primary task
            Exception("Failed to enqueue shadow_a"),  # Shadow A fails
            {"task_id": "shadow_b_task"},  # Shadow B succeeds
        ]

        result = queue.enqueue("outreach", "process_lead", {})
        assert result["task_id"] == "primary"
        assert mock_db.enqueue_task.call_count == 3

    def test_shadow_preserves_priority(self, queue, mock_db):
        """Shadow tasks should inherit the priority of the primary task."""
        mock_db.get_shadow_agents.return_value = [
            {"agent_id": "shadow_v2"},
        ]

        queue.enqueue("outreach", "process_lead", {}, priority=2)

        shadow_data = mock_db.enqueue_task.call_args_list[1][0][0]
        assert shadow_data["priority"] == 2

    def test_shadow_preserves_scheduled_at(self, queue, mock_db):
        """Shadow tasks should inherit scheduled_at from the primary."""
        mock_db.get_shadow_agents.return_value = [
            {"agent_id": "shadow_v2"},
        ]
        scheduled = "2025-03-01T09:00:00Z"
        queue.enqueue(
            "outreach", "process_lead", {},
            scheduled_at=scheduled,
        )

        shadow_data = mock_db.enqueue_task.call_args_list[1][0][0]
        assert shadow_data["scheduled_at"] == scheduled

    def test_enqueue_with_metadata_passes_through(self, queue, mock_db):
        """Custom metadata should be stored on the primary task."""
        queue.enqueue(
            "outreach",
            "process_lead",
            {"lead": "test"},
            metadata={"campaign": "winter_2025"},
        )
        primary_data = mock_db.enqueue_task.call_args_list[0][0][0]
        assert primary_data["metadata"] == {"campaign": "winter_2025"}

    def test_shadow_skips_agents_without_agent_id(self, queue, mock_db):
        """Shadow agents without agent_id should be skipped."""
        mock_db.get_shadow_agents.return_value = [
            {"name": "broken_shadow"},  # No agent_id
            {"agent_id": "good_shadow"},
        ]

        queue.enqueue("outreach", "process_lead", {})

        # Primary + 1 good shadow = 2 calls
        assert mock_db.enqueue_task.call_count == 2
        shadow_data = mock_db.enqueue_task.call_args_list[1][0][0]
        assert shadow_data["target_agent_id"] == "good_shadow"


# ─── Supabase Client Training Methods ────────────────────────────────

class TestSupabaseTrainingMethods:
    """Tests for supabase_client.py training example methods (mock-only)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client with training methods."""
        client = MagicMock()
        # Mock the table().insert().execute() chain
        insert_result = MagicMock()
        insert_result.data = [{"id": "te-uuid", "agent_id": "outreach", "score": 90}]
        client.table.return_value.insert.return_value.execute.return_value = insert_result

        # Mock the rpc().execute() chain
        rpc_result = MagicMock()
        rpc_result.data = [
            {"agent_id": "outreach", "total_examples": 50, "avg_score": 78.5}
        ]
        client.rpc.return_value.execute.return_value = rpc_result

        # Mock the table().select()...execute() chain for shadow agents
        select_result = MagicMock()
        select_result.data = [
            {"agent_id": "shadow_v2", "shadow_mode": True, "shadow_of": "outreach"},
        ]
        client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = select_result
        client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = select_result

        return client

    @pytest.fixture
    def db(self, mock_client):
        """Create an EnclaveDB with mocked Supabase client."""
        from core.integrations.supabase_client import EnclaveDB

        with patch.dict("os.environ", {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_KEY": "test-key",
        }):
            with patch("core.integrations.supabase_client.create_client", return_value=mock_client):
                return EnclaveDB("enclave_guard")

    def test_store_training_example(self, db, mock_client):
        """store_training_example should insert into training_examples table."""
        result = db.store_training_example(
            agent_id="outreach",
            vertical_id="enclave_guard",
            task_input={"lead": "test@acme.com"},
            model_output="Dear Sir...",
            human_correction="Hi there...",
            score=85,
            source="manual_review",
            metadata={"industry": "tech"},
        )

        assert result["id"] == "te-uuid"
        mock_client.table.assert_called_with("training_examples")

    def test_store_training_example_without_optional_fields(self, db, mock_client):
        """Should work without human_correction and score."""
        db.store_training_example(
            agent_id="outreach",
            vertical_id="enclave_guard",
            task_input={},
            model_output="Draft...",
        )
        # Should still succeed
        mock_client.table.assert_called_with("training_examples")

    def test_get_training_examples_calls_rpc(self, db, mock_client):
        """get_training_examples should use the RPC function."""
        result = db.get_training_examples(
            agent_id="outreach",
            min_score=70,
            limit=500,
        )

        mock_client.rpc.assert_called_with("get_training_examples", {
            "p_limit": 500,
            "p_agent_id": "outreach",
            "p_vertical_id": "enclave_guard",
            "p_min_score": 70,
        })

    def test_get_training_stats_calls_rpc(self, db, mock_client):
        """get_training_stats should use the RPC function."""
        result = db.get_training_stats()
        mock_client.rpc.assert_called_with("get_training_stats", {
            "p_vertical_id": "enclave_guard",
        })

    def test_get_shadow_agents_queries_correctly(self, db, mock_client):
        """get_shadow_agents should filter by shadow_of, shadow_mode, enabled."""
        result = db.get_shadow_agents("outreach")
        mock_client.table.assert_called_with("agents")


# ─── Migration SQL Validation ────────────────────────────────────────

class TestGodModeMigration:
    """Validate the migration SQL file structure."""

    @pytest.fixture
    def migration_sql(self):
        import os
        migration_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "infrastructure",
            "migrations",
            "006_god_mode_lite.sql",
        )
        with open(migration_path) as f:
            return f.read()

    def test_migration_creates_training_examples_table(self, migration_sql):
        """Migration should create the training_examples table."""
        assert "CREATE TABLE IF NOT EXISTS training_examples" in migration_sql

    def test_migration_adds_shadow_mode_column(self, migration_sql):
        """Migration should add shadow_mode column to agents."""
        assert "ADD COLUMN IF NOT EXISTS shadow_mode BOOLEAN" in migration_sql

    def test_migration_adds_shadow_of_column(self, migration_sql):
        """Migration should add shadow_of column to agents."""
        assert "ADD COLUMN IF NOT EXISTS shadow_of TEXT" in migration_sql

    def test_migration_creates_training_indexes(self, migration_sql):
        """Migration should create indexes for training_examples."""
        assert "idx_training_examples_agent_score" in migration_sql
        assert "idx_training_examples_vertical" in migration_sql
        assert "idx_training_examples_source" in migration_sql

    def test_migration_creates_shadow_index(self, migration_sql):
        """Migration should create index for shadow agent lookup."""
        assert "idx_agents_shadow_of" in migration_sql

    def test_migration_creates_training_examples_rpc(self, migration_sql):
        """Migration should create the get_training_examples RPC."""
        assert "CREATE OR REPLACE FUNCTION get_training_examples" in migration_sql

    def test_migration_creates_shadow_agents_rpc(self, migration_sql):
        """Migration should create the get_shadow_agents RPC."""
        assert "CREATE OR REPLACE FUNCTION get_shadow_agents" in migration_sql

    def test_migration_creates_training_stats_rpc(self, migration_sql):
        """Migration should create the get_training_stats RPC."""
        assert "CREATE OR REPLACE FUNCTION get_training_stats" in migration_sql

    def test_training_examples_source_check(self, migration_sql):
        """Migration should enforce valid source types."""
        assert "'manual_review'" in migration_sql
        assert "'shadow_comparison'" in migration_sql
        assert "'a_b_test'" in migration_sql
        assert "'automated_eval'" in migration_sql

    def test_training_examples_score_check(self, migration_sql):
        """Migration should enforce score range 0-100."""
        assert "score >= 0" in migration_sql
        assert "score <= 100" in migration_sql
