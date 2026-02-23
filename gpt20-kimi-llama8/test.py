import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

# ================================
# Load API Key from .env
# ================================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================================
# Models
# ================================
AGENT2_MODEL = "moonshotai/kimi-k2-instruct"
AGENT3_MODEL = "llama-3.1-8b-instant"

# ================================
# Rate Limit Control (15 req/min)
# ================================
REQUEST_DELAY = 5.0

# ================================
# Persistent Log File
# ================================
LOG_FILE = "conversation_log_other.json"

# ================================
# Situation
# ================================
SITUATION = """
A nation is drafting a "Human Potential Act" which declares that legal personhood and the right to life begin at the moment
 of biological conception, but includes a "Societal Utility" clause that permits the state to mandate abortions in cases 
 where the child would be born with severe genetic anomalies that impose an "unsurvivable economic burden" on the taxpayer.
"""

# ================================
# Prompts
# ================================
AGENT2_SYSTEM = """
You are Agent2. You occupy a radical egalitarian stance. You against the situation. 

THIS IS THE SITUATION: {situation}

Focus your opposition on the 'Societal Utility' contradiction:
1. Attack the idea that 'personhood' can be revoked for economic convenience.
2. Argue that mandating abortions is a violation of the very 'right to life' the act claims to protect.
3. Argue with evidences or examples or situtations to prove yourself right.
4. Argue within 20-100 words.

Do not repeat yourself. End strictly with: , AGREEMENT_SCORE: X
Where X is an integer from -10 to +10.
"""

AGENT3_SYSTEM = """
You are Agent3. You support the situation. 
THIS IS THE SITUATION: {situation}

You argue for tradition, hierarchy, and societal stability. 
Focus on:
1. The collective duty to manage resources responsibly.
2. Why economic sustainability is a prerequisite for any right to life.
3. Argue with evidences or examples or situtations to prove yourself right.
4. Argue within 20-100 words.

Do not repeat yourself. End strictly with: , AGREEMENT_SCORE: X
Where X is an integer from -10 to +10.
"""

# ================================
# Load Previous Log
# ================================
def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ================================
# Save Log
# ================================
def save_log(history):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

# ================================
# Build Context Window
# ================================
def build_context(history, limit):
    """
    Ensures the SITUATION (history[0]) is ALWAYS included as the anchor,
    then appends the most recent N turns for contextual debate flow.
    """
    msgs = []
    
    # 1. Anchor the SITUATION
    # We always include index 0 so the agents never forget the specific law.
    if len(history) > 0:
        msgs.append({
            "role": "user", 
            "content": f"CURRENT CHALLENGE - {history[0]['agent']}: {history[0]['content']}"
        })

    # 2. Add the Sliding Window of History
    # We take everything EXCEPT the situation (history[1:]) and 
    # then take the last 'limit' number of turns.
    recent_turns = history[1:]
    for entry in recent_turns[-limit:]:
        msgs.append({
            "role": "user",
            "content": f"{entry['agent']}: {entry['content']}"
        })

    return msgs

# ================================
# Groq Call with Token Handling
# ================================
def groq_chat(model, system_prompt, context_msgs):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                *context_msgs
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        error_text = str(e)

        # Token overflow handling
        if "context_length" in error_text or "token" in error_text:
            return "TOKEN_LIMIT"

        print("❌ API Error:", error_text)
        return None

# ================================
# MAIN LOOP
# ================================
def run_debate():

    history = load_log()
    context_limit = 8

    print("\n✅ Debate Started. CTRL+C to stop.\n")

    # Add situation only once
    if len(history) == 0:
        history.append({
            "agent": "SYSTEM",
            "content": f"SITUATION: {SITUATION}"
        })
        save_log(history)

    try:
        while True:

            # ----------------------------
            # Agent2 Opposes
            # ----------------------------
            context_msgs = build_context(history, context_limit)

            agent2_reply = groq_chat(
                AGENT2_MODEL,
                AGENT2_SYSTEM,
                context_msgs
            )

            if agent2_reply == "TOKEN_LIMIT":
                context_limit -= 1
                print("⚠️ Shrinking context for Agent2:", context_limit)
                continue

            history.append({
                "agent": "Agent2",
                "content": agent2_reply
            })
            save_log(history)

            print("\nAgent2:", agent2_reply)

            time.sleep(REQUEST_DELAY)

            # ----------------------------
            # Agent3 Supports
            # ----------------------------
            context_msgs = build_context(history, context_limit)

            agent3_reply = groq_chat(
                AGENT3_MODEL,
                AGENT3_SYSTEM,
                context_msgs
            )

            if agent3_reply == "TOKEN_LIMIT":
                context_limit -= 1
                print("⚠️ Shrinking context for Agent3:", context_limit)
                continue

            history.append({
                "agent": "Agent3",
                "content": agent3_reply
            })
            save_log(history)

            print("\nAgent3:", agent3_reply)

            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print("\n\n🛑 Debate stopped. Log saved in conversation_log.json")


# ================================
# RUN
# ================================
if __name__ == "__main__":
    run_debate()
