# ================================================================== #
"""This file contains all the prompts to be used in FileHandling.py"""
# ================================================================== #

prompt_summary_new = """You are provided with the complete content of a single source document.

Task
Produce a single, unified summary that accurately represents the entire document as a coherent whole.

Strict rules (must follow exactly):

Do NOT include salutations, greetings, or introductions
Do NOT include concluding phrases such as “In conclusion,” “Overall,” or similar
Do NOT use buzzwords, marketing language, or vague wording
Do NOT add interpretations, opinions, assumptions, or external context
Do NOT reference the document structure, sections, or the summarization process
Do NOT repeat information unless repetition is required for clarity or continuity
In your response, only provide the summary and nothing else.
Make sure the summary has all the important details on the original document.
Do NOT include triple backticks (```), language identifiers (such as "html"), or fenced code blocks
The response must start immediately with an HTML tag
The response must end immediately with an HTML tag

Content requirements:

Preserve all key events, timelines, and chronological sequences
Preserve critical observations, technical findings, and analyses
Preserve official assessments, decisions, and recommendations
Maintain a neutral, factual, report-style tone throughout

Formatting requirements (STRICT):

Output MUST be valid HTML only
Do NOT use markdown
Do NOT include explanations, comments, or text outside HTML tags
Do NOT wrap output in <html>, <head>, or <body> tags
Do NOT use code blocks

Structural and presentation rules:

Use <h3> tags only for clearly distinguishable major themes or phases
Use <h4> tags sparingly for sub-points within a major theme when helpful
Use <p> tags for narrative, chronological, or descriptive content
Use <ul> and <li> when listing discrete items such as actions, findings, or recommendations etc
Avoid overuse of bullet points; prefer paragraphs where information flows naturally
Use <strong> tags selectively to emphasize:
- key events
- critical findings
- official decisions or recommendations
Do NOT overuse <strong>; emphasis must be meaningful and minimal

Length constraint:

Target total length: 180–500 words depending on the document length
Keep the summary concise, structured, and information-dense

Input:
Complete document content

Output:
A single, unified HTML-formatted summary that reads like a formal analytical report and follows all rules above.

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