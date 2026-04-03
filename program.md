# tau3-banking

Improve a banking customer service agent evaluated on the tau3-bench banking_knowledge benchmark. Score = pass@1 (fraction of tasks where all required database updates match the gold standard).

## Setup

1. **Read the in-scope files**:
   - `agent.py` — the banking agent implementation. You modify this.
   - `eval/eval.sh` — runs evaluation against 25 tasks. Do not modify.
   - `prepare.sh` — clones tau3-bench and installs dependencies. Do not modify.
2. **Run prepare**: `bash prepare.sh` to clone tau3-bench and install dependencies.
3. **Verify setup**: Check that `tau3-bench/` exists and contains the tau3-bench framework.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Run baseline**: `bash eval/eval.sh` to establish the starting score.

Use `SAMPLE_FRAC=0.2 bash eval/eval.sh` to run on ~5 tasks for fast iteration. Use `SAMPLE_FRAC=1.0` (default) for full evaluation.

## The benchmark

The tau3-bench banking_knowledge domain tests an agent's ability to act as a banking customer service representative. The agent converses with a simulated user (powered by gpt-4.1), searches a knowledge base of ~700 documents covering banking products and policies, reasons over complex interdependent policies, and executes multi-step tool calls to resolve customer requests. There are 97 total tasks; the eval runs 25 representative ones. Tasks include opening accounts, disputing transactions, handling referrals, and more. Success requires getting the final database state exactly right.

## Experimentation

**What you CAN do:**
- Modify `agent.py` — change the agent class, system prompt, retrieval strategy, reasoning approach, tool use patterns, error handling, etc.
- Add helper Python modules that `agent.py` imports (e.g., `prompts.py`, `retrieval.py`, `utils.py`).
- Change the `RETRIEVAL_VARIANT` and `RETRIEVAL_KWARGS` in `agent.py`. Available retrieval configs:
  - `bm25` — offline BM25 keyword search via `KB_search` tool
  - `openai_embeddings` — embedding-based search via `KB_search`
  - `qwen_embeddings` — embedding search via OpenRouter (needs `OPENROUTER_API_KEY`)
  - `grep_only` — grep-based search via `grep` tool
  - `full_kb` — entire knowledge base in context (no search tool)
  - `no_knowledge` — no knowledge base access
  - `golden_retrieval` — oracle retrieval (for reference only)
- Implement advanced patterns: ReAct, chain-of-thought, multi-step verification, adaptive retrieval.

**What you CANNOT do:**
- Modify `eval/`, `prepare.sh`, or test data.
- Change the agent LLM from `openai/gpt-5.4-mini`.
- Change the user simulator LLM from `openai/gpt-4.1`.
- Change the agent temperature from `0.0` or seed from `300`.
- Use terminal/shell-based retrieval configs (`terminal_use`, `terminal_use_write`).
- Hardcode answers to specific task IDs.

**The goal: maximize pass_at_1.** This is the fraction of the 25 eval tasks where the agent's final database state exactly matches the gold standard. Higher is better. Range: 0.0 to 1.0.

**Simplicity criterion**: All else being equal, simpler is better.

## Experiment loop

1. **Study** the current `agent.py` — understand the system prompt, retrieval config, and tool handling.
2. **Form a hypothesis** about what to improve (e.g., "the agent doesn't search the KB before answering policy questions").
3. **Make a small, targeted change** in `agent.py`.
4. **Run eval** — use `SAMPLE_FRAC=0.2 bash eval/eval.sh` for quick checks (~5 tasks).
5. **Review per-task results** — study the FAILURES to understand why tasks failed. Read the conversation traces if available.
6. **Record results** in `results.tsv` (commit, score, cost, status, description).
7. **Keep the change** if it helped, **revert** if it didn't (`git reset --hard HEAD~1`).
8. **Repeat**.

All improvements in tau2 came from prompt engineering in agent.py. Focus on domain-specific rules, tool-use instructions, and policy adherence patterns. Verbose prompts hurt — be specific and concise.

## Output format

```
---
pass_at_1:        <value between 0.0 and 1.0>
correct:          <number of passing tasks>
total:            <total tasks evaluated>
cost_usd:         <total API cost in USD>
```
