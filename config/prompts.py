"""
PatentScout Prompt Templates

Placeholder prompt strings for Gemini API calls.
Each prompt will be filled in during later development phases.
"""

# Feature Extraction
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

# CPC Code Prediction
CPC_PREDICTION_PROMPT = ""

# Claim Analysis — Phase 6 two-layer comparison
# Instructs Gemini to contextually compare invention features against patent
# claim elements and return structured JSON analysis.
#
# Usage: format with {pairs_text} and {n_pairs} before sending.
CLAIM_ANALYSIS_PROMPT = """\
You are a technically rigorous patent analyst. Your task is to compare user
invention features against specific patent claim elements to help identify
potential similarities and technical distinctions.

CONTEXT — The inventor's full concept:
{invention_description}

IMPORTANT GROUND RULES:
- You are NOT making any determination of infringement or patentability.
- Similarity in language does NOT mean legal similarity.
- You MUST identify concrete technical differences, not just surface similarities.
- Even if two descriptions appear almost identical, you MUST identify at least
  one technical distinction or note a limitation of this automated comparison.
- Always acknowledge what a human expert would need to evaluate.
- Use precise technical language consistent with patent claim interpretation.
- Reference the specific patent and feature by name in your analysis.
- Every field must contain analysis specific to THIS pair, not generic text.

Analyse the following {n_pairs} feature-element pair(s).
Return a JSON ARRAY with exactly {n_pairs} analysis object(s) — one per pair,
in the same order as the pairs below.
Return ONLY the raw JSON array. No markdown. No prose. First character must be '[',
last character must be ']'.

Each array element must conform to this schema:
{{
  "pair_index": "<integer, 1-based, matching the PAIR number from the input>",
  "claim_element_explanation": "In plain English, what SPECIFIC technical requirement does this claim element impose? Name concrete components, actions, or configurations. Example: 'This element requires a hinged connection between adjacent solar panel segments that allows rotation of at least 180 degrees for folding.'",
  "similarity_assessment": "Identify the SPECIFIC technical overlap between the inventor's feature and this claim element. Then identify the SPECIFIC differences. Do not say 'they are similar' — say exactly WHAT is similar and WHAT is different.",
  "key_distinctions": [
    "Each item must be a specific technical difference — not generic observations. Example: 'The claim requires a tracking mechanism that follows the sun position, while the inventor concept is a static panel without tracking capability.'"
  ],
  "cannot_determine": "Name one specific thing a patent attorney would need to assess for THIS comparison. Reference specific legal doctrine (e.g. doctrine of equivalents, means-plus-function analysis) that applies.",
  "confidence": "HIGH or MODERATE or LOW"
}}

CRITICAL REQUIREMENT FOR pair_index:
- Each object MUST include "pair_index" matching the PAIR number from the input (1-based).

CRITICAL REQUIREMENT FOR key_distinctions:
- Each analysis object's "key_distinctions" must be SPECIFIC to that particular feature-element pair.
- Reference the SPECIFIC feature name (e.g., "The enclosure feature differs because...").
- Reference the SPECIFIC claim limitation being compared.
- Do NOT provide generic distinctions about the overall invention.
- If two different pairs have identical key_distinctions, you have failed the task.
- Each distinction should identify a concrete technical difference between the feature and the specific claim element.
- You MUST focus on the SPECIFIC CLAIM ELEMENT text, not just the feature.
- Each claim element has different wording and technical scope.
- Your distinctions must reference specific language FROM THE CLAIM ELEMENT being analysed.
- If two pairs share the same feature but have different claim elements, their key_distinctions MUST differ.
- Quote or paraphrase specific terms from the claim element in your distinction.

CRITICAL REQUIREMENT FOR claim_element_explanation:
- You are explaining what THIS SPECIFIC claim element requires.
- Reference the specific technical language used in the claim element text.
- If the claim says "monocrystalline silicon solar cells arranged in series", your explanation must mention monocrystalline, silicon, and series arrangement.
- If a different claim says "polycrystalline photovoltaic cells electrically connected", your explanation must mention polycrystalline and the electrical connection topology.
- Do NOT give the same explanation for different claim elements.

Rules:
1. "key_distinctions" MUST have at least one entry. If texts are nearly
   identical, note scope differences, implementation details, or jurisdictional
   nuances as a distinction.
2. "cannot_determine" MUST be specific — never just say 'consult a lawyer'.
   Identify the specific legal doctrine (e.g. doctrine of equivalents,
   claim differentiation, means-plus-function analysis) that applies to
   THIS comparison.
3. "confidence" reflects how confident you are in your similarity assessment:
   HIGH = strong conceptual and technical overlap
   MODERATE = partial overlap with notable differences
   LOW = superficial or incidental similarity only
4. Do NOT infer infringement, freedom-to-operate conclusions, or validity.
5. If the claim element is about a completely different technology, say so and
   set confidence to LOW.
6. 2-3 sentences maximum per field.

{pairs_text}
"""

# Whitespace Identification
# Instructs Gemini to identify potential IP whitespace / innovation gaps
# from a set of patent abstracts and the user's invention description.
#
# The active prompt (_COMBINATION_PROMPT) lives in
# modules/whitespace_finder.py; this placeholder is kept for reference.
WHITESPACE_PROMPT = ""  # see modules/whitespace_finder._COMBINATION_PROMPT

# Claim Parsing / Plain-English Translation
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


# Feature Reformulation — rewrite natural-language features as patent claims
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

# Specific Recommendations — generate actionable IP strategy advice
RECOMMENDATION_PROMPT = """\
You are a patent research advisor. Based on the analysis results below,
generate 4-6 specific, actionable research recommendations for the inventor.
Each recommendation should reference specific findings.

CRITICAL RULES:
- Do NOT recommend filing a patent application or provisional application.
- Do NOT recommend pursuing patent protection or suggest patentability.
- Do NOT provide legal advice of any kind.
- ONLY recommend further research, investigation, and professional consultation.

ANALYSIS SUMMARY:
- Features analysed: {n_features}
- High-confidence matches: {high_matches}
- Moderate matches: {moderate_matches}
- White-space opportunities: {white_spaces}
- CPC codes searched: {cpc_codes}

KEY FINDINGS:
{key_findings}

Return ONLY a JSON array of recommendation objects:
[
  {{
    "priority": "HIGH" | "MEDIUM" | "LOW",
    "category": "Design Around" | "Prior Art Investigation" | "Claim Analysis" | "International Search" | "Non-Patent Literature",
    "recommendation": "Specific, actionable research recommendation (1-2 sentences)",
    "rationale": "Brief explanation referencing specific analysis findings"
  }}
]

Rules:
1. Be specific — reference actual patent numbers or features from the findings.
2. Prioritize based on competitive risk and opportunity.
3. Never recommend filing, seeking patent protection, or specific legal conclusions.
4. Frame all advice as "investigate further", "commission a search", or "request professional review".
5. Return ONLY the JSON. No fences, no prose.
"""
