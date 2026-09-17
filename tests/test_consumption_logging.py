import asyncio
import importlib
import json
import logging
from types import SimpleNamespace

import pytest

from app.agent import agent as agent_module
from app.agent import graph as graph_module


def _flush(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def _close_file_handlers(*loggers: logging.Logger) -> None:
    for logger in loggers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def _json_lines(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_jsonl_logger_reopens_when_its_file_is_deleted(tmp_path):
    telemetry = importlib.import_module("app.agent.telemetry")
    log_path = tmp_path / "node_output.log"
    output_logger, _ = telemetry.build_jsonl_logger(
        "deleted_node_output", str(log_path)
    )

    try:
        telemetry.write_jsonl(output_logger, {"node": "first"})
        _flush(output_logger)
        log_path.unlink()

        telemetry.write_jsonl(output_logger, {"node": "second"})
        _flush(output_logger)

        assert log_path.exists()
        assert _json_lines(log_path) == [{"node": "second"}]
    finally:
        _close_file_handlers(output_logger)


def test_node_usage_schema_includes_zero_usage_nodes_and_aggregates_llm_calls():
    summarize = getattr(agent_module, "_summarize_node_usage")
    records = summarize(
        ["load_call", "infer_behavioral_evaluation", "finalize"],
        [
            {
                "node": "infer_behavioral_evaluation",
                "provider": "openrouter",
                "model": "model-a",
                "requests": 1,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cost_usd": 0.001,
                "cost_complete": True,
            },
            {
                "node": "infer_behavioral_evaluation",
                "provider": "openrouter",
                "model": "model-a",
                "requests": 2,
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
                "cost_usd": 0.002,
                "cost_complete": True,
            },
        ],
        default_provider="openrouter",
        default_model="model-a",
    )

    assert records == [
        {
            "node": "load_call",
            "sequence": 1,
            "execution_count": 1,
            "provider": None,
            "model": None,
            "request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "known_cost_usd": 0.0,
            "cost_complete": True,
            "uses_llm": False,
        },
        {
            "node": "infer_behavioral_evaluation",
            "sequence": 2,
            "execution_count": 1,
            "provider": "openrouter",
            "model": "model-a",
            "request_count": 3,
            "prompt_tokens": 150,
            "completion_tokens": 30,
            "total_tokens": 180,
            "cost_usd": 0.003,
            "known_cost_usd": 0.003,
            "cost_complete": True,
            "uses_llm": True,
        },
        {
            "node": "finalize",
            "sequence": 3,
            "execution_count": 1,
            "provider": None,
            "model": None,
            "request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "known_cost_usd": 0.0,
            "cost_complete": True,
            "uses_llm": False,
        },
    ]


def test_analyze_writes_separate_node_and_overall_consumption_files(monkeypatch, tmp_path):
    node_path = tmp_path / "node_consumption.log"
    overall_path = tmp_path / "overall_consumption.log"
    monkeypatch.setattr(agent_module.settings, "NODE_CONSUMPTION_LOG_PATH", str(node_path))
    monkeypatch.setattr(agent_module.settings, "OVERALL_CONSUMPTION_LOG_PATH", str(overall_path))

    node_logger, configured_node_path = agent_module._build_consumption_logger(
        "node_consumption", agent_module.settings.NODE_CONSUMPTION_LOG_PATH
    )
    overall_logger, configured_overall_path = agent_module._build_consumption_logger(
        "overall_consumption", agent_module.settings.OVERALL_CONSUMPTION_LOG_PATH
    )

    result = object()

    class FakeGraph:
        async def ainvoke(self, initial_state):
            telemetry = importlib.import_module("app.agent.telemetry")
            for node_name in (
                "load_call",
                "criteria_ready",
                "infer_behavioral_evaluation",
                "finalize",
            ):
                telemetry.record_node_execution(node_name)
            return {
                "result": result,
                "node_trace": ["load_call", "infer_behavioral_evaluation", "finalize"],
                "usage_list": [
                    {
                        "node": "infer_behavioral_evaluation",
                        "provider": "openrouter",
                        "model": "model-a",
                        "requests": 1,
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "total_tokens": 15,
                        "cost_usd": None,
                        "cost_complete": False,
                    }
                ],
            }

    qa_agent = agent_module.QAAgent.__new__(agent_module.QAAgent)
    qa_agent.llm_client = SimpleNamespace(provider="openrouter", model="model-a")
    qa_agent._graph = FakeGraph()
    qa_agent._node_consumption_logger = node_logger
    qa_agent._node_consumption_log_path = configured_node_path
    qa_agent._overall_consumption_logger = overall_logger
    qa_agent._overall_consumption_log_path = configured_overall_path

    try:
        returned = asyncio.run(
            qa_agent.analyze(SimpleNamespace(call_id="call-1"))
        )
        _flush(node_logger)
        _flush(overall_logger)

        assert returned is result
        node_records = _json_lines(node_path)
        overall_records = _json_lines(overall_path)
        assert [record["node"] for record in node_records] == [
            "load_call",
            "criteria_ready",
            "infer_behavioral_evaluation",
            "finalize",
        ]
        assert node_records[0]["total_tokens"] == 0
        assert node_records[2]["total_tokens"] == 15
        assert len(overall_records) == 1
        assert overall_records[0]["call_id"] == "call-1"
        assert overall_records[0]["total_tokens"] == 15
        assert overall_records[0]["node_count"] == 4
        assert overall_records[0]["llm_node_count"] == 1
        assert "nodes" not in overall_records[0]
    finally:
        _close_file_handlers(node_logger, overall_logger)


def test_node_output_wrapper_logs_sync_async_and_error_outputs(monkeypatch, tmp_path):
    telemetry = importlib.import_module("app.agent.telemetry")
    output_path = tmp_path / "node_output.log"
    monkeypatch.setattr(graph_module.settings, "NODE_OUTPUT_LOG_PATH", str(output_path))
    output_logger, configured_path = telemetry.build_jsonl_logger(
        "node_output", graph_module.settings.NODE_OUTPUT_LOG_PATH
    )
    assert configured_path == output_path

    def sync_node(state):
        return {"sync_value": 1}

    async def async_node(state):
        return {"async_value": 2}

    def failing_node(state):
        raise RuntimeError("node exploded")

    def none_node(state):
        return None

    wrapped_sync = graph_module._with_node_output_logging(
        "sync_node", sync_node, output_logger
    )
    wrapped_async = graph_module._with_node_output_logging(
        "async_node", async_node, output_logger
    )
    wrapped_failing = graph_module._with_node_output_logging(
        "failing_node", failing_node, output_logger
    )
    wrapped_none = graph_module._with_node_output_logging(
        "none_node", none_node, output_logger
    )

    tracking_token = telemetry.begin_node_execution_tracking()
    try:
        assert asyncio.run(wrapped_sync({"call": SimpleNamespace(call_id="call-2")})) == {
            "sync_value": 1
        }
        assert asyncio.run(wrapped_async({"call": SimpleNamespace(call_id="call-2")})) == {
            "async_value": 2
        }
        assert asyncio.run(wrapped_none({"call": SimpleNamespace(call_id="call-2")})) is None
        with pytest.raises(RuntimeError, match="node exploded"):
            asyncio.run(wrapped_failing({"call": SimpleNamespace(call_id="call-2")}))
        assert telemetry.end_node_execution_tracking(tracking_token) == [
            "sync_node", "async_node", "none_node", "failing_node"
        ]
        _flush(output_logger)

        records = _json_lines(output_path)
        assert records[0]["node"] == "sync_node"
        assert records[0]["status"] == "success"
        assert records[0]["output"] == {"sync_value": 1}
        assert records[1]["node"] == "async_node"
        assert records[1]["output"] == {"async_value": 2}
        assert records[2]["node"] == "none_node"
        assert records[2]["output"] is None
        assert records[3]["node"] == "failing_node"
        assert records[3]["status"] == "error"
        assert records[3]["error"] == "node exploded"
    finally:
        if telemetry.current_node_executions() is not None:
            telemetry.end_node_execution_tracking(tracking_token)
        _close_file_handlers(output_logger)
