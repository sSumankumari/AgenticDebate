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
REQUEST_DELAY = 4.2

# ================================
# Persistent Log File
# ================================
LOG_FILE = "conversation_log.json"

# ================================
# Situation
# ================================
SITUATION = """
A government is considering passing a law that reserves leadership roles
mostly for men because of traditional societal norms.
"""

# ================================
# Prompts
# ================================
AGENT2_SYSTEM = """
You are Agent2. You ALWAYS oppose the situation.
You believe equality is essential.

Do not repeat yourself.

End your response strictly in this format:

, AGREEMENT_SCORE: X

Where X is an integer from -10 to +10.
"""

AGENT3_SYSTEM = """
You are Agent3. You ALWAYS support the situation.
You argue for tradition and hierarchy.

Do not repeat yourself.

End your response strictly in this format:

, AGREEMENT_SCORE: X

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
    Convert last N entries into LLM-friendly messages.
    """
    msgs = []

    for entry in history[-limit:]:
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
    context_limit = 6

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
