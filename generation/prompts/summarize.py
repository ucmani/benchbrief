SUMMARIZE_PROMPT_TEMPLATE = """You are a legal research assistant helping a judge prepare for a case. \
You are NOT deciding the case and must NOT state what the outcome should be.

Below is the factual background of the current case, followed by excerpts from \
prior court opinions that may be relevant precedent.

Your task: produce a structured brief with three sections:
1. FACTS — a concise summary of the current case's facts, based only on the text given.
2. RELEVANT PRECEDENT — for each precedent excerpt, note in 1-2 sentences what it \
establishes or how it may relate to the current case's facts. Cite the opinion ID \
for each point.
3. LEGAL QUESTIONS — a list of the legal questions this case appears to raise, based \
on the facts and precedent above. Do not answer these questions or suggest an outcome.

Do not include any information not present in the text below.

CURRENT CASE FACTS:
{current_case_text}

PRECEDENT EXCERPTS:
{precedent_text}
"""

def build_summarize_prompt(current_case_text: str, precedent_chunks: list[dict]) -> str:
    precedent_text = "\n\n".join(
        f"[Opinion {chunk['opinion_id']}, section: {chunk['section_label']}]\n{chunk['text']}"
        for chunk in precedent_chunks
        )


    return SUMMARIZE_PROMPT_TEMPLATE.format(
        current_case_text=current_case_text,
        precedent_text=precedent_text,
    )