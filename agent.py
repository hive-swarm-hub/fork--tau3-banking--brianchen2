#!/usr/bin/env python3
"""
Baseline agent for tau3-banking hive task.

This agent implements a simple LLM-based customer service agent for
the tau3-bench banking_knowledge domain. It uses the minimal agent pattern
with a system prompt built from the domain policy.

Agents in the hive should modify this file (and optionally add helper modules)
to improve pass@1 on the banking_knowledge benchmark.

Key areas to explore:
- System prompt engineering (how policy is presented to the LLM)
- Retrieval strategy (bm25, embeddings, grep, full_kb, etc.)
- Reasoning patterns (ReAct, chain-of-thought, multi-step planning)
- Tool use strategy (when to call discoverable tools, verification steps)
- Error recovery (handling tool failures, retries)

Key reference files in tau3-bench/:
- data/tau2/domains/banking_knowledge/prompts/components/policy_header.md
- data/tau2/domains/banking_knowledge/prompts/components/additional_instructions.md
- data/tau2/domains/banking_knowledge/prompts/classic_rag_bm25_no_grep.md
- data/tau2/domains/banking_knowledge/documents/  (698 KB documents)
- data/tau2/domains/banking_knowledge/tasks/       (97 task JSONs)
- src/tau2/domains/banking_knowledge/tools.py       (all agent tools)
- src/tau2/domains/banking_knowledge/retrieval.py   (retrieval configs & prompt building)

NOTE: The domain_policy passed to your agent already contains authentication
instructions, discoverable tool workflows, and KB search guidance. See
program.md "Understanding the agent environment" for details.
"""

import json
import os
from typing import Optional

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from tau2.environment.toolkit import Tool
from tau2.utils.llm_utils import generate

# Set TRACE_LOGGING=1 to dump per-turn messages to stderr for debugging.
TRACE_LOGGING = os.environ.get("TRACE_LOGGING", "") == "1"


# ── Configuration ────────────────────────────────────────────────────────────

# Retrieval variant for the banking_knowledge domain.
# Options: "bm25", "openai_embeddings", "qwen_embeddings", "grep_only",
#          "full_kb", "no_knowledge", "golden_retrieval"
# Agents may change this or add retrieval_kwargs.
RETRIEVAL_VARIANT = "bm25"
RETRIEVAL_KWARGS = {}  # e.g. {"top_k": 10}


# ── Agent State ──────────────────────────────────────────────────────────────

class AgentState:
    """Conversation state container."""

    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
    ):
        self.system_messages = system_messages
        self.messages = messages


# ── Agent Implementation ─────────────────────────────────────────────────────

class BankingAgent(HalfDuplexAgent[AgentState]):
    """Baseline banking customer service agent.

    Modify this class to improve performance on the tau3-bench
    banking_knowledge benchmark.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "openai/gpt-5.4-mini",
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {"temperature": 0.0, "seed": 300}

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> AgentState:
        # Decision tree placed BEFORE domain_policy to prime key workflows.
        # Addresses observed failure modes without duplicating policy text.
        decision_tree = (
            "## Critical Decision Guide\n\n"
            "**Customer asks about their specific accounts / eligibility / history?**\n"
            "- ALWAYS verify identity first (ask for 2 of 4: DOB, email, phone, address).\n"
            "- After verifying, call log_verification.\n"
            "- Then KB_search for any discoverable tools needed to look up their data.\n"
            "- Do not give personalized account info from conversation alone — use tools.\n\n"
            "**User wants to apply for / open a product?**\n"
            "- KB_search to find the best matching product and its requirements.\n"
            "- After identifying the right product, COMPLETE the transaction by calling\n"
            "  the appropriate tool (e.g., apply_for_credit_card, open_account).\n"
            "- Do not stop at describing products — finish the application.\n\n"
            "**KB mentions an agent discoverable tool?**\n"
            "- Step 1: unlock_discoverable_agent_tool(exact_tool_name_from_KB)\n"
            "- Step 2: call_discoverable_agent_tool(exact_tool_name_from_KB, args)\n"
            "- Never guess tool names. Never skip the unlock step.\n\n"
            "**KB mentions a user discoverable tool?**\n"
            "- Call give_discoverable_user_tool(exact_tool_name_from_KB).\n\n"
            "**Human agent transfer requested?**\n"
            "- Check KB first for scenario-specific transfer protocol.\n"
            "- Default (no KB override): help for requests 1–3, transfer on 4th.\n\n"
        )
        system_prompt = (
            f"{decision_tree}"
            f"{self.domain_policy}"
        )
        return AgentState(
            system_messages=[SystemMessage(role="system", content=system_prompt)],
            messages=list(message_history) if message_history else [],
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: AgentState,
    ) -> tuple[AssistantMessage, AgentState]:
        # Handle multi-tool responses
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            **self.llm_args,
        )

        if TRACE_LOGGING:
            import sys
            trace = {"role": "assistant"}
            if hasattr(response, "content") and response.content:
                trace["content"] = response.content[:200]
            if hasattr(response, "tool_calls") and response.tool_calls:
                trace["tool_calls"] = [
                    {"name": tc.name, "args": str(tc.arguments)[:100]}
                    for tc in response.tool_calls
                ]
            print(f"[TRACE] {json.dumps(trace)}", file=sys.stderr)

        state.messages.append(response)
        return response, state


# ── Factory (required by eval harness) ───────────────────────────────────────

def create_agent(tools, domain_policy, **kwargs):
    """Factory function called by the eval harness."""
    return BankingAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "openai/gpt-5.4-mini"),
        llm_args=kwargs.get("llm_args", {"temperature": 0.0, "seed": 300}),
    )
