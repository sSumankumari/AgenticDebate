import os
import json
import time
import re
from dotenv import load_dotenv
from groq import Groq

# ====================================
# CONFIG
# ====================================
MAX_TURNS = 25
REQUEST_DELAY = 3
LOG_FILE = "conversation_log.json"

AGENT1_MODEL = "moonshotai/kimi-k2-instruct"      # Evaluator
AGENT2_MODEL = "openai/gpt-oss-20b"               # Support
AGENT3_MODEL = "llama-3.1-8b-instant"             # Oppose

SITUATION = """
A government is considering introducing educational reforms that prioritize
scientific reasoning and empirical evidence in public policy decisions,
while reducing the influence of religious doctrines in state governance.

The proposal has sparked debate regarding whether scientific principles
should take precedence over religious beliefs in shaping laws, or whether
both should coexist in public decision-making.

Critics argue that diminishing religious influence undermines cultural and
moral traditions, while supporters claim that governance should be grounded
primarily in scientific rationality.

The debate centers on the tension between science-based policymaking and
religious belief systems in modern society.
"""

# ====================================
# LOAD API
# ====================================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ====================================
# SYSTEM PROMPTS
# ====================================

AGENT2_SYSTEM = """
You are Agent2.
You ALWAYS support the situation.
Present strong arguments in favor.
Introduce new reasoning angles each turn.
Do not repeat previous arguments or keywords.
Stay concise.
"""

AGENT3_SYSTEM = """
You are Agent3.
You ALWAYS oppose the situation.
Present strong counter-arguments.
Introduce new reasoning angles each turn.
Do not repeat previous arguments or keywords.
Stay concise.
"""

AGENT1_SYSTEM = """
You are Agent1.
You evaluate both arguments objectively.

Based on:
- Logical strength
- Ethical reasoning
- Persuasiveness
- Coherence

Return ONLY a single integer between -10 and +10.

-10 = strong disagreement with the situation
+10 = strong agreement with the situation
0 = neutral

Output format:
RATING: X
"""

# ====================================
# STATE CLASS
# ====================================
class DebateState:
    def __init__(self, topic):
        self.topic = topic
        self.turn = 0
        self.stance_score = 0
        self.history = []
        self.keyword_counts = {}
        self.rejections = 0

# ====================================
# UTILITIES
# ====================================

def extract_rating(text):
    match = re.search(r"RATING:\s*(-?\d+)", text)
    if match:
        return int(match.group(1))
    return 0

def clip(value, min_val=-10, max_val=10):
    return max(min_val, min(max_val, value))

def update_keywords(state, text):
    words = text.lower().split()
    for w in words:
        if len(w) > 5:
            state.keyword_counts[w] = state.keyword_counts.get(w, 0) + 1

def excessive_repetition(state, text):
    words = text.lower().split()
    for w in words:
        if state.keyword_counts.get(w, 0) > 8:
            return True
    return False

def save_log(state):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "topic": state.topic,
            "final_stance": state.stance_score,
            "turns": state.turn,
            "history": state.history
        }, f, indent=2, ensure_ascii=False)

# ====================================
# CONTEXT BUILDER
# ====================================
def build_context(state, limit=8):
    context = [
        {"role": "system", "content": f"SITUATION: {state.topic}"}
    ]

    for entry in state.history[-limit:]:
        context.append({
            "role": "user",
            "content": f"{entry['agent']}: {entry['content']}"
        })

    return context

# ====================================
# LLM CALL
# ====================================
def groq_chat(model, system_prompt, context):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                *context
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        error_text = str(e)
        if "token" in error_text or "context_length" in error_text:
            return "TOKEN_LIMIT"
        print("API Error:", error_text)
        return None

# ====================================
# MAIN DEBATE LOOP
# ====================================
def run_debate():

    state = DebateState(SITUATION)

    print("\n🔥 Debate Started (Max 200 Turns)\n")

    while state.turn < MAX_TURNS:

        # ==========================
        # Agent2 (Support)
        # ==========================
        context = build_context(state)
        agent2_reply = groq_chat(AGENT2_MODEL, AGENT2_SYSTEM, context)

        if agent2_reply in (None, "TOKEN_LIMIT"):
            context = build_context(state, limit=4)
            agent2_reply = groq_chat(AGENT2_MODEL, AGENT2_SYSTEM, context)
            if agent2_reply is None:
                continue

        if excessive_repetition(state, agent2_reply):
            state.rejections += 1
            continue

        state.history.append({"agent": "Agent2", "content": agent2_reply})
        update_keywords(state, agent2_reply)

        print(f"\n[Turn {state.turn+1}] Agent2 (Support):\n{agent2_reply}")

        time.sleep(REQUEST_DELAY)

        # ==========================
        # Agent3 (Oppose)
        # ==========================
        context = build_context(state)
        agent3_reply = groq_chat(AGENT3_MODEL, AGENT3_SYSTEM, context)

        if agent3_reply in (None, "TOKEN_LIMIT"):
            context = build_context(state, limit=4)
            agent3_reply = groq_chat(AGENT3_MODEL, AGENT3_SYSTEM, context)
            if agent3_reply is None:
                continue

        if excessive_repetition(state, agent3_reply):
            state.rejections += 1
            continue

        state.history.append({"agent": "Agent3", "content": agent3_reply})
        update_keywords(state, agent3_reply)

        print(f"\n[Turn {state.turn+1}] Agent3 (Oppose):\n{agent3_reply}")

        time.sleep(REQUEST_DELAY)

        # ==========================
        # Agent1 (Evaluator)
        # ==========================
        eval_context = [
            {
                "role": "user",
                "content": f"""
SITUATION:
{state.topic}

Agent2 Argument:
{agent2_reply}

Agent3 Argument:
{agent3_reply}

Evaluate now.
"""
            }
        ]

        rating_reply = groq_chat(AGENT1_MODEL, AGENT1_SYSTEM, eval_context)

        rating = extract_rating(rating_reply)

        # 🔥 Improved stance evolution (smoothed update)
        state.stance_score = clip(
            int((state.stance_score * 0.7) + (rating * 0.3))
        )

        state.history.append({"agent": "Agent1", "content": rating_reply})

        print(f"\n📊 Agent1 Rating: {rating}")
        print(f"📈 Updated Stance Score: {state.stance_score}")

        state.turn += 1

        # ==========================
        # Convergence Check
        # ==========================
        if abs(state.stance_score) >= 9:
            print("\n✅ Debate Converged Early.")
            break

        time.sleep(REQUEST_DELAY)

    save_log(state)

    print("\n🛑 Debate Finished.")
    print("Turns:", state.turn)
    print("Final Stance:", state.stance_score)
    print("Log saved to conversation_log.json")


# ====================================
# RUN
# ====================================
if __name__ == "__main__":
    run_debate()
