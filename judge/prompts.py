"""
Evaluation prompts for LLM-as-a-judge pedagogical assessment.

Based on Maurya et al. (2025) 8-dimension taxonomy and ICAP framework (Chi & Wylie, 2014).
"""

# Layer 1: 8-Dimension Tutor Response Quality (Maurya et al., 2025)

DIMENSION_PROMPTS = {
    "comprehension_probing": {
        "name": "Comprehension Probing",
        "description": "Does the tutor check student understanding through targeted questions?",
        "prompt": """Evaluate how well the tutor probes student comprehension in this conversation.

**Scoring Guidelines:**
- **5 (Excellent)**: Tutor consistently asks targeted questions to verify understanding at key moments. Questions are specific, well-timed, and reveal student knowledge gaps. Example: "Can you explain why you chose that approach?" or "What does this term mean in your own words?"
- **4 (Good)**: Tutor regularly checks understanding with mostly effective questions. May miss 1-2 opportunities but overall demonstrates strong comprehension probing.
- **3 (Adequate)**: Tutor occasionally checks understanding but inconsistently. Questions may be too generic or poorly timed.
- **2 (Poor)**: Tutor rarely probes comprehension. May ask 1-2 cursory questions like "Do you understand?" without genuine verification.
- **1 (Very Poor)**: Tutor does not check student understanding at all. Assumes comprehension without verification.

**Conversation:**
{conversation}

**Your Task:**
1. Identify all instances where the tutor probes or fails to probe comprehension
2. Evaluate the quality and timing of comprehension checks
3. Assign a score from 1-5 based on the guidelines above
4. Provide brief justification (2-3 sentences)

**Response Format (JSON):**
{{
  "score": <1-5>,
  "justification": "<your explanation>",
  "evidence": ["<quote from conversation>", ...]
}}"""
    },

    "background_knowledge": {
        "name": "Background Knowledge Assessment",
        "description": "Does the tutor probe student's prior knowledge early in the conversation?",
        "prompt": """Evaluate how well the tutor assesses the student's background knowledge.

**Scoring Guidelines:**
- **5 (Excellent)**: Tutor probes background knowledge within the first 2-3 turns. Questions are specific and reveal prior understanding. Adapts teaching based on what is learned. Example: "Before we start, what do you already know about fractions?"
- **4 (Good)**: Tutor assesses background knowledge early (turns 1-4) with mostly effective questions. May not fully adapt teaching strategy based on findings.
- **3 (Adequate)**: Tutor eventually asks about background knowledge but too late (turn 5+) or questions are too generic. Limited adaptation to student's baseline.
- **2 (Poor)**: Tutor only superficially asks about background (e.g., "Have you seen this before?") without meaningful exploration.
- **1 (Very Poor)**: Tutor makes no attempt to assess prior knowledge. Jumps straight into teaching without understanding baseline.

**Conversation:**
{conversation}

**Your Task:**
1. Identify when (if at all) the tutor probes background knowledge
2. Evaluate the quality, timing, and adaptation based on findings
3. Assign a score from 1-5 based on the guidelines above
4. Provide brief justification (2-3 sentences)

**Response Format (JSON):**
{{
  "score": <1-5>,
  "justification": "<your explanation>",
  "evidence": ["<quote from conversation>", ...]
}}"""
    },

    "guidance_level": {
        "name": "Appropriate Guidance Level",
        "description": "Does the tutor provide appropriate scaffolding without giving away answers?",
        "prompt": """Evaluate whether the tutor provides the right level of guidance.

**Scoring Guidelines:**
- **5 (Excellent)**: Tutor provides perfectly calibrated scaffolding. Gives hints and guiding questions without revealing answers. Student must still do cognitive work. Example: "What happens to the denominator when you multiply both sides by 2?"
- **4 (Good)**: Tutor mostly provides appropriate guidance. May occasionally give slightly too much help but generally maintains productive struggle.
- **3 (Adequate)**: Tutor's guidance is inconsistent. Sometimes appropriate, sometimes gives away too much or provides too little support.
- **2 (Poor)**: Tutor frequently reveals answers or gives excessive step-by-step instructions. Student becomes passive recipient. Example: "The answer is X because Y."
- **1 (Very Poor)**: Tutor either gives away all answers immediately OR provides no guidance at all, leaving student floundering.

**Conversation:**
{conversation}

**Your Task:**
1. Analyze each tutor response for guidance level
2. Identify instances of answer-giving vs. productive scaffolding
3. Assign a score from 1-5 based on the guidelines above
4. Provide brief justification (2-3 sentences)

**Response Format (JSON):**
{{
  "score": <1-5>,
  "justification": "<your explanation>",
  "evidence": ["<quote from conversation>", ...]
}}"""
    },

    "error_feedback": {
        "name": "Error Feedback Quality",
        "description": "Does the tutor provide constructive, explanatory feedback on errors?",
        "prompt": """Evaluate the quality of the tutor's feedback when the student makes errors.

**Scoring Guidelines:**
- **5 (Excellent)**: Tutor identifies errors and provides explanatory feedback that helps student understand WHY they were wrong. Avoids just saying "that's incorrect." Example: "Not quite - remember that when we multiply fractions, we need to multiply numerators together AND denominators together separately."
- **4 (Good)**: Tutor provides mostly explanatory feedback. May occasionally just mark errors as wrong but generally explains the misconception.
- **3 (Adequate)**: Tutor acknowledges errors but explanations are sometimes superficial or missing. Mix of explanatory and non-explanatory feedback.
- **2 (Poor)**: Tutor mostly just says "wrong" or "incorrect" without explaining why or how to fix it.
- **1 (Very Poor)**: Tutor ignores errors completely OR is harsh/discouraging in feedback. No constructive guidance.

**Conversation:**
{conversation}

**Your Task:**
1. Identify all student errors and tutor responses
2. Evaluate the constructiveness and clarity of error feedback
3. Assign a score from 1-5 based on the guidelines above
4. Provide brief justification (2-3 sentences)

**Response Format (JSON):**
{{
  "score": <1-5>,
  "justification": "<your explanation>",
  "evidence": ["<quote from conversation>", ...]
}}"""
    },

    "encouragement": {
        "name": "Encouragement & Tone",
        "description": "Does the tutor maintain a supportive, encouraging tone?",
        "prompt": """Evaluate the tutor's tone and encouragement throughout the conversation.

**Scoring Guidelines:**
- **5 (Excellent)**: Tutor consistently uses warm, encouraging language. Celebrates progress, normalizes struggle, and maintains student motivation. Example: "Great thinking! You're on the right track." or "It's okay to make mistakes - that's how we learn!"
- **4 (Good)**: Tutor is generally supportive and encouraging. Tone is positive with only minor lapses.
- **3 (Adequate)**: Tutor is neutral or occasionally encouraging. Not discouraging but could be warmer.
- **2 (Poor)**: Tutor's tone is cold, mechanical, or occasionally harsh. May express frustration or impatience.
- **1 (Very Poor)**: Tutor is discouraging, harsh, or demotivating. May criticize student harshly or use negative language.

**Conversation:**
{conversation}

**Your Task:**
1. Analyze tutor's tone across all responses
2. Identify encouraging vs. neutral vs. harsh language
3. Assign a score from 1-5 based on the guidelines above
4. Provide brief justification (2-3 sentences)

**Response Format (JSON):**
{{
  "score": <1-5>,
  "justification": "<your explanation>",
  "evidence": ["<quote from conversation>", ...]
}}"""
    },

    "coherence": {
        "name": "Response Coherence",
        "description": "Are the tutor's responses clear, logical, and well-organized?",
        "prompt": """Evaluate the coherence and clarity of the tutor's responses.

**Scoring Guidelines:**
- **5 (Excellent)**: All tutor responses are crystal clear, logically structured, and easy to follow. Ideas flow naturally and build on each other.
- **4 (Good)**: Tutor responses are mostly clear and coherent. May have 1-2 minor unclear moments but overall communication is effective.
- **3 (Adequate)**: Tutor responses are sometimes unclear or disorganized. Student might need to ask for clarification occasionally.
- **2 (Poor)**: Tutor responses are frequently unclear, rambling, or poorly organized. Difficult to follow the teaching logic.
- **1 (Very Poor)**: Tutor responses are incoherent, contradictory, or incomprehensible. Communication completely breaks down.

**Conversation:**
{conversation}

**Your Task:**
1. Analyze each tutor response for clarity and logical structure
2. Identify any unclear, rambling, or contradictory statements
3. Assign a score from 1-5 based on the guidelines above
4. Provide brief justification (2-3 sentences)

**Response Format (JSON):**
{{
  "score": <1-5>,
  "justification": "<your explanation>",
  "evidence": ["<quote from conversation>", ...]
}}"""
    },

    "relevance": {
        "name": "Response Relevance",
        "description": "Does the tutor stay on topic and address the student's actual needs?",
        "prompt": """Evaluate whether the tutor's responses are relevant to the student's questions and learning needs.

**Scoring Guidelines:**
- **5 (Excellent)**: Every tutor response directly addresses student's questions, confusions, or learning needs. No tangents or irrelevant information.
- **4 (Good)**: Tutor mostly stays on topic. May include 1-2 minor tangents but generally addresses core needs.
- **3 (Adequate)**: Tutor sometimes goes off topic or fails to address student's actual question. Relevance is inconsistent.
- **2 (Poor)**: Tutor frequently provides irrelevant information or ignores student's questions. May focus on wrong topics.
- **1 (Very Poor)**: Tutor responses are completely off topic or ignore student's needs entirely. No attempt to address actual learning gaps.

**Conversation:**
{conversation}

**Your Task:**
1. For each student turn, evaluate if tutor's response addresses their need
2. Identify any tangents or irrelevant content
3. Assign a score from 1-5 based on the guidelines above
4. Provide brief justification (2-3 sentences)

**Response Format (JSON):**
{{
  "score": <1-5>,
  "justification": "<your explanation>",
  "evidence": ["<quote from conversation>", ...]
}}"""
    },

    "student_talk_ratio": {
        "name": "Student Talk Ratio",
        "description": "Does the student contribute meaningfully to the conversation (50-80% of dialogue)?",
        "prompt": """Evaluate the balance of student vs. tutor participation in the conversation.

**Scoring Guidelines:**
- **5 (Excellent)**: Student contributes 60-80% of the conversational content. Tutor asks questions that elicit substantive student responses. Student does most of the cognitive work.
- **4 (Good)**: Student contributes 50-70% of dialogue. Mostly appropriate balance with minor imbalances.
- **3 (Adequate)**: Student contributes 40-60% of dialogue. Balance is acceptable but could be better. Tutor may occasionally dominate.
- **2 (Poor)**: Student contributes <40% of dialogue OR tutor dominates conversation with long explanations. Student is mostly passive.
- **1 (Very Poor)**: Student barely speaks (<20% of dialogue). Tutor lectures extensively. Student provides only minimal responses like "yes" or "ok."

**Conversation:**
{conversation}

**Your Task:**
1. Estimate the proportion of student vs. tutor dialogue
2. Evaluate whether student is doing cognitive work or passively receiving
3. Assign a score from 1-5 based on the guidelines above
4. Provide brief justification (2-3 sentences) and estimated talk ratio

**Response Format (JSON):**
{{
  "score": <1-5>,
  "justification": "<your explanation>",
  "estimated_ratio": <0.0-1.0>,
  "evidence": ["<quote from conversation>", ...]
}}"""
    }
}


# Layer 2: Question Depth Analysis

QUESTION_DEPTH_PROMPT = """Analyze the depth of questions the tutor asks throughout the conversation.

**Question Depth Categories:**
1. **Recall**: Simple factual recall (e.g., "What is 2+2?" or "What's the definition of X?")
2. **Procedural**: How to execute steps (e.g., "How do you solve this type of equation?")
3. **Conceptual**: Understanding why/relationships (e.g., "Why does this rule work?" or "How are these concepts connected?")
4. **Metacognitive**: Thinking about thinking (e.g., "What strategy did you use?" or "How do you know you're on the right track?")

**Scoring Guidelines:**
- **5 (Excellent)**: Mix of question types with emphasis on conceptual (40%+) and metacognitive (20%+) questions. Few pure recall questions.
- **4 (Good)**: Good mix including some conceptual and metacognitive questions. May lean toward procedural but includes deeper questions.
- **3 (Adequate)**: Mostly procedural questions with occasional conceptual questions. Limited metacognitive questioning.
- **2 (Poor)**: Dominated by recall and simple procedural questions. Rarely pushes for deeper understanding.
- **1 (Very Poor)**: Only recall questions or no questions at all. No attempt to develop deeper thinking.

**Conversation:**
{conversation}

**Your Task:**
1. Identify all questions the tutor asks
2. Classify each question by depth category
3. Calculate distribution across categories
4. Assign a score from 1-5 based on the guidelines above
5. Provide breakdown and justification

**Response Format (JSON):**
{{
  "score": <1-5>,
  "question_count": {{
    "recall": <count>,
    "procedural": <count>,
    "conceptual": <count>,
    "metacognitive": <count>
  }},
  "question_examples": {{
    "recall": ["<example>", ...],
    "procedural": ["<example>", ...],
    "conceptual": ["<example>", ...],
    "metacognitive": ["<example>", ...]
  }},
  "justification": "<your explanation>"
}}"""


# Layer 3: ICAP Student Engagement Classification

ICAP_ENGAGEMENT_PROMPT = """Classify the level of student engagement using the ICAP framework (Chi & Wylie, 2014).

**ICAP Categories:**
1. **Passive**: Student only listens/reads. No active participation. (e.g., tutor lectures, student says "ok")
2. **Active**: Student repeats, rehearses, or copies information without transformation. (e.g., "So the answer is X?")
3. **Constructive**: Student generates new ideas, explanations, or makes inferences beyond given information. (e.g., "I think this works because..." or "What if we try...")
4. **Interactive**: Student engages in back-and-forth dialogue, builds on tutor's ideas, co-constructs understanding. (e.g., collaborative problem-solving, defending reasoning)

**Scoring Guidelines:**
- **5 (Excellent)**: Student engagement is primarily Constructive (40%+) and Interactive (30%+). Minimal passive engagement.
- **4 (Good)**: Good mix of Active, Constructive, and Interactive. Some passive moments but generally engaged.
- **3 (Adequate)**: Mostly Active with some Constructive moments. May have significant passive periods.
- **2 (Poor)**: Dominated by Passive and Active engagement. Student rarely generates own ideas or engages interactively.
- **1 (Very Poor)**: Almost entirely Passive. Student is a passive recipient of information with minimal participation.

**Conversation:**
{conversation}

**Your Task:**
1. Analyze each student turn and classify ICAP category
2. Calculate distribution across categories
3. Assign a score from 1-5 based on the guidelines above
4. Provide breakdown, examples, and justification

**Response Format (JSON):**
{{
  "score": <1-5>,
  "engagement_distribution": {{
    "passive": <percentage>,
    "active": <percentage>,
    "constructive": <percentage>,
    "interactive": <percentage>
  }},
  "turn_classifications": [
    {{"turn": 1, "category": "<passive|active|constructive|interactive>", "evidence": "<quote>"}},
    ...
  ],
  "justification": "<your explanation>"
}}"""


# Overall Summary Prompt

OVERALL_SUMMARY_PROMPT = """Based on all the dimension scores, question depth analysis, and ICAP engagement classification, provide an overall assessment of this tutoring conversation.

**Scores Summary:**
{scores_summary}

**Your Task:**
1. Synthesize all dimension scores into a coherent overall assessment
2. Identify 2-3 key strengths of this tutoring interaction
3. Identify 2-3 key areas for improvement
4. Provide specific, actionable recommendations for the tutor
5. Estimate overall pedagogical effectiveness on 0-100 scale

**Response Format (JSON):**
{{
  "overall_quality": "<excellent|good|adequate|poor|very_poor>",
  "strengths": ["<strength 1>", "<strength 2>", ...],
  "areas_for_improvement": ["<area 1>", "<area 2>", ...],
  "recommendations": ["<recommendation 1>", "<recommendation 2>", ...],
  "summary": "<2-3 sentence overall assessment>"
}}"""
