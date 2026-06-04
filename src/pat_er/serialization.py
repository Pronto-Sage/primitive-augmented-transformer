from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence


HERMES_TOOL_CALL_START = "<tool_call>"
HERMES_TOOL_CALL_END = "</tool_call>"


@dataclass(frozen=True)
class PATERToolCall:
    """OpenAI-compatible function call payload carried in Hermes tags."""

    name: str
    arguments: dict[str, Any]

    def to_openai_tool_call(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": canonical_json(self.arguments),
            },
        }


def canonical_json(value: Any) -> str:
    """Compact deterministic JSON for tool schemas, evidence, and tool calls."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_tools_block(tools: Sequence[Mapping[str, Any]]) -> str:
    """Render tool definitions in a PAT-ER control block.

    Each tool definition should be OpenAI-style:
    {"type": "function", "function": {"name": ..., "parameters": ...}}
    or a compact {"name": ..., "parameters": ...} mapping.
    """

    rendered_tools = "\n".join(f"<tool>\n{canonical_json(tool)}\n</tool>" for tool in tools)
    return f"<tools>\n{rendered_tools}\n</tools>"


def render_evidence_block(evidence: Sequence[Mapping[str, Any]]) -> str:
    rendered = "\n".join(canonical_json(item) for item in evidence)
    return f"<evidence>\n{rendered}\n</evidence>"


def render_formula(formula: str) -> str:
    return f"<formula>\n{formula}\n</formula>"


def render_event_graph(events: Sequence[Mapping[str, Any]]) -> str:
    rendered = "\n".join(canonical_json(event) for event in events)
    return f"<event_graph>\n{rendered}\n</event_graph>"


def render_hermes_tool_call(name: str, arguments: Mapping[str, Any]) -> str:
    """Render one Hermes/vLLM-style tool call.

    vLLM's Hermes parser extracts JSON enclosed in <tool_call> tags and expects
    fields named "name" and "arguments".
    """

    payload = canonical_json({"name": name, "arguments": dict(arguments)})
    return f"{HERMES_TOOL_CALL_START}\n{payload}\n{HERMES_TOOL_CALL_END}"


def parse_hermes_tool_calls(text: str) -> list[PATERToolCall]:
    """Parse one or more Hermes-style tool calls from model text."""

    pattern = re.compile(
        re.escape(HERMES_TOOL_CALL_START) + r"(.*?)" + re.escape(HERMES_TOOL_CALL_END),
        flags=re.DOTALL,
    )
    calls: list[PATERToolCall] = []
    for match in pattern.finditer(text):
        payload = json.loads(match.group(1).strip())
        if not isinstance(payload, dict):
            raise ValueError("tool call payload must be a JSON object")
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call payload must contain a non-empty string name")
        if not isinstance(arguments, dict):
            raise ValueError("tool call payload must contain an arguments object")
        calls.append(PATERToolCall(name=name, arguments=arguments))
    return calls


def render_pater_prompt(
    *,
    text: str,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    formula: str | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    output_prelude: str = "",
) -> str:
    """Render an end-to-end PAT-ER training/inference record."""

    parts = ["<pat_er>", "<text>", text, "</text>"]
    if evidence:
        parts.append(render_evidence_block(evidence))
    if formula:
        parts.append(render_formula(formula))
    if events:
        parts.append(render_event_graph(events))
    if tools:
        parts.append(render_tools_block(tools))
    parts.extend(["<output>", output_prelude, "</output>", "</pat_er>"])
    return "\n".join(part for part in parts if part != "")


def build_reference_release_tool() -> dict[str, Any]:
    """Small OpenAI-style tool schema used by smoke scripts."""

    return {
        "type": "function",
        "function": {
            "name": "verify_release",
            "description": "Verify that a release candidate has passing tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "requires": {"type": "string"},
                },
                "required": ["run_id", "requires"],
            },
        },
    }


def build_reference_pater_record() -> str:
    """Full PAT-ER record exercising evidence, formula, roles, primitives, tools."""

    return render_pater_prompt(
        text="rule if tests_passed implies release_ok evidence e_2 tests_passed",
        evidence=[
            {
                "id": "e_2",
                "source": "ci_log",
                "text": "tests_passed",
                "reliability": 0.97,
            }
        ],
        formula="<atom> tests_passed </atom> IMPLIES <atom> release_ok </atom>",
        events=[
            {
                "event": "<evt:0>",
                "predicate": "verify_release",
                "arguments": [
                    {
                        "slot": "<arg:0>",
                        "span": "tests_passed",
                        "proto_roles": ["<role:source>", "<role:causes_change>"],
                    }
                ],
                "primitive": "<prim:modus_ponens>",
                "support": "<support:belief>",
            }
        ],
        tools=[build_reference_release_tool()],
        output_prelude="<needs_verification>",
    )
