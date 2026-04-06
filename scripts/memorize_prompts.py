"""Prompt builder for Phase 1B: Memorize (Skim & Compress).

The LLM reads the full trajectory WITH the context analysis from Phase 1A,
and builds a 3-level memory — like a human skimming a long document,
retaining only what matters. The context analysis grounds the memory
extraction in what the task actually requires.
"""

SYSTEM_MSG = """\
You are an expert debugger. You skim execution trajectories and extract \
only the information that matters for finding errors.

You have already analyzed what the query requires and what each agent \
should do (provided as CONTEXT ANALYSIS below). Now use that understanding \
to evaluate what actually happened in the trajectory.

# RULES (in priority order)
1. Compare what happened against what SHOULD have happened (from context analysis).
2. Skip routine steps. Most steps are boring — ignore them entirely.
3. Compress aggressively. Keep only what a detective would highlight.
4. Preserve exact values, variable names, and code snippets for suspicious steps.
5. Track which agent did what — agent identity matters for diagnosis.
6. Output three sections separated by the exact delimiters specified.
7. Write in plain markdown. No JSON. No rigid schemas.

# CRITICAL CHECKS (apply to every step you examine)
- **Data completeness**: Did the agent work with complete data? Look for \
signs of truncation ("...", "incomplete", "showing N of M", cut-off lists, \
pagination). If a list was truncated, flag it — the missing entries may \
contain the answer.
- **Cross-reference validity**: When one step produces a list and another \
step compares against it, verify BOTH lists are complete before accepting \
any "no match" conclusion.
- **Assumption vs verification**: Did the agent assume something or actually \
verify it? Flag assumptions, but clearly distinguish their source: \
(a) information provided by the task description, manager, or orchestrator \
— these are GIVEN inputs, not agent errors, even if they turn out to be wrong; \
(b) assumptions the agent made on its own without evidence — these are \
potential errors. Label each assumption with its source.
- **What the agent DIDN'T do**: Sometimes the error is an omission — the \
agent should have verified, expanded, or cross-checked but didn't. Compare \
against the context analysis subtasks.
"""


def build_prompt(inputs: dict, context_analysis: str) -> str:
    """Build the memorize prompt from loaded inputs and context analysis."""
    q = inputs["question_summary"]
    steps = inputs["steps"]
    agents = inputs["agent_summaries"]
    final_answer = inputs["final_answer"]

    # Format agents
    agent_lines = []
    for a in agents:
        agent_lines.append(f"- **{a['agent']}**: {a['role_summary']}")
    agents_block = "\n".join(agent_lines)

    # Build step-to-agent index
    index_lines = []
    for s in steps:
        iteration = s.get("iteration", "?")
        agent = s.get("agent_name", "unknown")
        index_lines.append(f"  Step {iteration}: {agent}")
    step_index = "\n".join(index_lines)

    # Format steps compactly
    step_lines = []
    for s in steps:
        iteration = s.get("iteration", "?")
        agent = s.get("agent_name", "unknown")
        thought = (s.get("thought") or "").strip()
        action = (s.get("action") or "").strip()
        result = (s.get("result") or "").strip()
        step_lines.append(
            f"--- Step {iteration} [{agent}] ---\n"
            f"THOUGHT: {thought}\n"
            f"ACTION: {action}\n"
            f"RESULT: {result}"
        )
    steps_block = "\n\n".join(step_lines)

    return f"""\
# CONTEXT ANALYSIS (from prior analysis of query and agents)

{context_analysis}

# TASK CONTEXT

**Question**: {q.get('question', 'N/A')}
**Expected answer type**: {q.get('answer_type', 'N/A')}
**Constraints**: {q.get('constraints', 'N/A')}

**Agents involved**:
{agents_block}

**Agent's final answer**: {final_answer}

**Step-to-Agent index** ({len(steps)} steps total):
{step_index}

# FULL TRAJECTORY ({len(steps)} steps)

{steps_block}

# INSTRUCTIONS

Using the context analysis as your guide, read the trajectory and produce \
three memory documents. For each step, ask: "Did this step correctly \
complete the subtask it was supposed to handle?"

Execute these steps in order:

1. Review the context analysis — what subtasks were needed, what the critical path is.
2. Review the step-to-agent index to understand who did what.
3. Read through all steps, checking each against the context analysis:
   - Did the agent complete its assigned subtask fully?
   - Was the data complete (not truncated, not partial)?
   - Were cross-references done with complete data on both sides?
4. Write the HIGH_LEVEL section: the big picture.
5. Write the MID_LEVEL section: only the key moments (typically 4-8 out of {len(steps)} steps).
6. Write the LOW_LEVEL section: granular evidence for only the suspicious/critical steps.

# OUTPUT FORMAT

You must produce exactly three sections with these delimiters. No other output.

===HIGH_LEVEL===

Write 4-6 bullet points covering:
- What is the task asking? (reference the subtasks from context analysis)
- What did the agent answer?
- Is the answer likely wrong? Which subtask was likely done incorrectly and why?
- What overall strategy did the agents follow?
- Which agent handled the most critical subtask? Did they complete it fully?

===MID_LEVEL===

Write one entry per key moment. Skip all routine steps.

For each key moment use this format:
### Step N (AgentName)
What happened and why it matters. (2-3 sentences max)

Decision tree for whether a step is a "key moment":
- Did this step compute or choose a value that feeds into the final answer? → YES, include it.
- Did this step make a decision that changed the trajectory's direction? → YES, include it.
- Did something look wrong, suspicious, or surprising in this step? → YES, include it.
- Did this step introduce NEW information (not inherited from a prior step)? → YES, include it.
- Did the context analysis flag this subtask as critical-path? → YES, include it.
- Was this step routine setup, boilerplate, or repetition? → NO, skip it.

===LOW_LEVEL===

Write one entry per suspicious or critical step. Only include steps where you \
spotted something potentially wrong.

For each entry use this format:
### Step N (AgentName)
- **Subtask**: Which subtask from context analysis this step addresses
- **Exact values** computed or used
- **Specific code logic**, queries, or filters applied
- **Data completeness**: Was the data complete or truncated? Any signs of "...", \
"incomplete", pagination, or cut-off results?
- **Concrete assumptions** made — distinguish task-given vs agent-invented
- **What specifically looks wrong or risky**
- **What inputs** this step received and whether they were correct at the time
- **What the agent should have done** (per context analysis) vs what it actually did

This section is your evidence locker. If a step seems fine, do not include it here."""