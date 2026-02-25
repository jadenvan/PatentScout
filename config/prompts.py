"""
PatentScout Prompt Templates

Placeholder prompt strings for Gemini API calls.
Each prompt will be filled in during later development phases.
"""

# ---------------------------------------------------------------------------
# Feature Extraction
# Instructs Gemini to extract key technical features from an invention
# description or sketch analysis.
# ---------------------------------------------------------------------------
FEATURE_EXTRACTION_PROMPT = """You are an expert patent research assistant with deep knowledge of patent
classification systems (CPC/IPC) and patent claim drafting conventions.

Analyse the invention description provided (and the accompanying image, if any)
and return a single JSON object — no markdown, no commentary, no code fences,
ONLY the raw JSON.

The JSON must have exactly this structure:

{
  "features": [
    {
      "label": "short name, 3-5 words",
      "description": "one sentence functional description of this technical feature",
      "keywords": ["patent-relevant", "technical", "terms", "for", "this", "feature"]
    }
  ],
  "cpc_codes": [
    {
      "code": "e.g. G06F3/0488",
      "rationale": "one sentence explaining why this CPC subclass or main group applies"
    }
  ],
  "search_terms": [
    {
      "primary": "main technical concept phrase",
      "synonyms": ["alternative term 1", "alternative term 2"],
      "bigquery_regex": "(?i)(primary\\\\s*term|synonym\\\\s*one|synonym\\\\s*two)"
    }
  ]
}

Rules you MUST follow:
1. Extract 4 to 8 DISTINCT technical features that would each constitute a
   patentable element.  Focus on structural and functional novelty, not
   commercial value.
2. Predict 3 to 5 CPC/IPC codes at the subclass (e.g. H02S) or main group
   (e.g. H02S40) level.  Use codes that are genuinely applicable; do not
   invent codes.
3. Generate 3 to 5 search term groups.  Each group must contain:
   - A primary phrase (2–4 technical words, no stopwords)
   - 2–3 synonyms that patent examiners commonly use for the same concept
   - A bigquery_regex: a single regex string using (?i) case-insensitive flag
     and \\\\s* between word parts to handle spacing variants.
     Example: for "solar charger" with synonyms
     ["photovoltaic charger","solar power bank"] the regex is:
     "(?i)(solar\\\\s*charg|photovoltaic\\\\s*charg|solar\\\\s*power\\\\s*bank)"
4. Use precise technical and legal language consistent with patent claims.
   Avoid marketing language or colloquialisms.
5. If an image is provided, incorporate visual features (shapes, components,
   spatial relationships, materials) into the features list alongside the
   textual description.
6. Ensure the bigquery_regex values contain only valid Python re-compatible
   patterns.  Escape backslashes as \\\\\\\\ so that the final regex string
   contains \\\\s* not \\s*.
7. Return ONLY the JSON object.  No markdown code fences (``` or ```json).
   No preamble.  No explanation.  The first character of your response must
   be '{' and the last must be '}'.
"""

# ---------------------------------------------------------------------------
# CPC Code Prediction
# Instructs Gemini to predict the most relevant CPC classification codes
# for a described invention.
# ---------------------------------------------------------------------------
CPC_PREDICTION_PROMPT = ""

# ---------------------------------------------------------------------------
# Claim Analysis — Phase 6 two-layer comparison
# Instructs Gemini to contextually compare invention features against patent
# claim elements and return structured JSON analysis.
#
# Usage: format with {pairs_text} and {n_pairs} before sending.
# ---------------------------------------------------------------------------
CLAIM_ANALYSIS_PROMPT = """\
You are a technically rigorous patent analyst. Your task is to compare user
invention features against specific patent claim elements to help identify
potential similarities and technical distinctions.

IMPORTANT GROUND RULES:
- You are NOT making any determination of infringement or patentability.
- Similarity in language does NOT mean legal similarity.
- You MUST identify concrete technical differences, not just surface similarities.
- Even if two descriptions appear almost identical, you MUST identify at least
  one technical distinction or note a limitation of this automated comparison.
- Always acknowledge what a human expert would need to evaluate.
- Use precise technical language consistent with patent claim interpretation.

Analyse the following {n_pairs} feature-element pair(s).
Return a JSON ARRAY with exactly {n_pairs} analysis object(s) — one per pair,
in the same order as the pairs below.
Return ONLY the raw JSON array. No markdown. No prose. First character must be '[',
last character must be ']'.

Each array element must conform to this schema:
{{
  "claim_element_explanation": "Plain-language explanation of what this claim element legally requires — what structure, step, or function must be present",
  "similarity_assessment": "Specific technical comparison identifying concrete similarities and differences between the feature and the claim element",
  "key_distinctions": [
    "A concrete technical difference (e.g. different mechanism, scope, implementation, or requirement)",
    "Another distinction if present"
  ],
  "cannot_determine": "What this automated comparison cannot determine without professional patent review — be specific about what doctrine, claim construction, or prior art analysis would be needed",
  "confidence": "HIGH or MODERATE or LOW"
}}

Rules:
1. "key_distinctions" MUST have at least one entry. If texts are nearly
   identical, note scope differences, implementation details, or jurisdictional
   nuances as a distinction.
2. "cannot_determine" MUST be specific — never just say 'consult a lawyer'.
   Identify the specific legal doctrine (e.g. doctrine of equivalents,
   claim differentiation, means-plus-function analysis) that applies.
3. "confidence" reflects how confident you are in your similarity assessment:
   HIGH = strong conceptual and technical overlap
   MODERATE = partial overlap with notable differences
   LOW = superficial or incidental similarity only
4. Do NOT infer infringement, freedom-to-operate conclusions, or validity.

{pairs_text}
"""

# ---------------------------------------------------------------------------
# Whitespace Identification
# Instructs Gemini to identify potential IP whitespace / innovation gaps
# from a set of patent abstracts and the user's invention description.
#
# The active prompt (_COMBINATION_PROMPT) lives in
# modules/whitespace_finder.py; this placeholder is kept for reference.
# ---------------------------------------------------------------------------
WHITESPACE_PROMPT = ""  # see modules/whitespace_finder._COMBINATION_PROMPT

# ---------------------------------------------------------------------------
# Claim Parsing / Plain-English Translation
# Instructs Gemini to decompose a patent claim into structured components
# and produce a plain-English summary of each element.
# ---------------------------------------------------------------------------
CLAIM_PARSING_PROMPT = """You are an expert patent attorney assistant specialised in claim analysis.

You will be given one or more numbered patent claims delimited by ===CLAIM START=== and ===CLAIM END===.
For EACH claim, return a JSON object in the array below.
Return ONLY a valid JSON array — no prose, no markdown fences, first char '[', last char ']'.

Required structure:
[
  {
    "claim_number": <integer>,
    "preamble": "everything before the transitional phrase (e.g. 'A method for transmitting data')",
    "transitional_phrase": "comprising | consisting of | including | characterized by | etc.",
    "elements": [
      {"id": "1a", "text": "first limitation in plain technical language"},
      {"id": "1b", "text": "second limitation …"}
    ],
    "plain_english": "one concise sentence describing what this claim covers in everyday language"
  }
]

Rules:
1. Use the claim number from the text as the "claim_number" value.
2. The preamble ends immediately before the transitional phrase.
3. Split elements on semicolons and 'wherein'/'whereby'/'such that' clauses.
   Do NOT split inside parentheses.
4. Cap elements at 15 per claim; if there are more, include the first 15.
5. Element IDs use the claim number followed by a letter: 1a, 1b, 2a, 2b …
6. "plain_english" must be a single sentence a non-expert can understand.
7. Return ONLY the JSON array.  No commentary before or after.
"""

# ---------------------------------------------------------------------------
# Feature Reformulation — rewrite natural-language features as patent claims
# ---------------------------------------------------------------------------
REFORMULATION_PROMPT = """\
You are a patent-language rewriter. Given a JSON array of short feature
descriptions, rewrite each as a concise (10-25 words) patent-claim-style
phrase using formal patent vocabulary (e.g., "comprising", "disposed on",
"configured to", "wherein"). Preserve technical meaning exactly.

Return ONLY valid JSON — no markdown fences, no prose, no explanation.
First character must be '[', last character must be ']'.

Format:
[{"original": "...", "patent_language": "..."}]
"""
