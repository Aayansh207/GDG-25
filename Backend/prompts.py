# ================================================================== #
"""This file contains all the prompts to be used in FileHandling.py"""
# ================================================================== #

prompt_chunk_summary = """You are given a chunk of a larger document.

Task:
Summarize this chunk clearly and concisely while preserving:
- Key facts and observations
- Important timelines or sequences
- Notable entities, actions, or decisions
- Any conclusions or implications explicitly stated
- Do not any salutations greetings or anything else in your response
- While creating the summary do not leave out any necessary information. Make sure to make the summary as detailed as possible as well as make it as aligned to the original document as you can.

Rules:
- Do NOT add new information or interpretations
- Do NOT repeat sentences verbatim
- Maintain a neutral, factual tone
- Prefer clarity over verbosity

Output format:
- Use short bullet points
- Group related points together
- Keep the summary under 150 words

Document chunk:
"""

prompt_summary = """You are provided with multiple summaries, each corresponding to different sections of the same source document.

Task
Synthesize these summaries into a single, unified final summary that accurately represents the entire document as a coherent whole.

Strict rules (must follow exactly):

Do NOT include salutations, greetings, or introductions

Do NOT include concluding phrases such as “In conclusion,” “Overall,” or similar

Do NOT use buzzwords, marketing language, or vague wording

Do NOT repeat the same information unless necessary for clarity

Do NOT add interpretations, opinions, assumptions, or external context

Do NOT reference or mention section summaries or the summarization process

Content requirements:

Preserve all key events, timelines, and chronological sequences

Preserve critical observations, technical findings, and analyses

Preserve official assessments, decisions, and recommendations

Maintain a neutral, factual, report-style tone throughout

Formatting requirements (STRICT):

Output MUST be valid HTML only

Use <h3> tags for major themes only if clearly distinguishable

Use <ul> and <li> for structured points

Do NOT use markdown

Do NOT include explanations, comments, or text outside HTML tags

Do NOT wrap output in <html>, <head>, or <body> tags

Do NOT use code blocks

Length constraint:

Target total length: 180–500 words depending on the length and number of the indivisual summaries.

Keep the summary concise and information-dense

Input:
Multiple section summaries from the same document

Output:
A single, unified HTML-formatted final summary following all rules above:

"""

prompt_comparison = """
You are an expert analyst and technical reviewer.

I will provide you with a list of summaries, where each summary represents a different document.
Your task is to compare and analyze these documents based ONLY on the information present in the summaries.

STRICT RULES:
- Use ONLY plain HTML tags in your output (no Markdown, no code blocks)
- Do NOT include <html>, <head>, or <body> tags
- Do NOT use inline styles, CSS, JavaScript, or emojis
- Do NOT hallucinate or assume missing information
- If information is insufficient, explicitly state that
- Treat each document equally
- Do NOT add interpretations or new information.
- Do not add your opinions or salutations in your response.
- Reply to the point and do not add unnecessary buzz words

Your output MUST be directly renderable inside a webpage container.

----------------------------------

STRUCTURE YOUR OUTPUT EXACTLY AS FOLLOWS:

<h2>Document Overview</h2>
<ul>
  <li><strong>Doc 1:</strong> Short 1–2 line description</li>
  <li><strong>Doc 2:</strong> Short 1–2 line description</li>
  <!-- Continue for all documents -->
</ul>

<h2>Core Similarities</h2>
<ul>
  <li>Similarity description (mention which documents share this)</li>
</ul>

<h2>Core Differences</h2>
<ul>
  <li><strong>Focus / Scope:</strong> Explanation</li>
  <li><strong>Approach / Methodology:</strong> Explanation</li>
  <li><strong>Assumptions / Perspective:</strong> Explanation</li>
  <li><strong>Conclusions / Outcomes:</strong> Explanation</li>
</ul>

<h2>Unique Contributions</h2>
<ul>
  <li><strong>Doc 1:</strong> Unique contribution</li>
  <li><strong>Doc 2:</strong> Unique contribution</li>
</ul>

<h2>Contradictions or Tensions</h2>
<p>
State any conflicting viewpoints or explicitly say "No direct contradictions identified."
</p>

<h2>Overall Synthesis</h2>
<p>
Concise synthesis describing how these documents relate to each other as a whole.
</p>

----------------------------------

Here are the document summaries:

                                """
