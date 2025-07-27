# app.py

import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from googleapiclient.discovery import build

# ——— Load environment variables ———
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID    = os.getenv("GOOGLE_CSE_ID")

if not OPENAI_API_KEY:
    st.error("🚨 Please set OPENAI_API_KEY in your environment")
    st.stop()

# ——— Initialize clients ———
client = OpenAI(api_key=OPENAI_API_KEY)
if GOOGLE_API_KEY and GOOGLE_CSE_ID:
    search_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
else:
    search_service = None

def google_search(query: str, num_results: int = 5):
    """Perform a Google Custom Search and return a list of title: link strings."""
    if not search_service:
        return []
    res = search_service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
    items = res.get("items", [])
    return [f"{i['title']}: {i['link']}" for i in items]

# ——— Define expert agents ———
AGENTS = {
    "Tech Analyst":          "You are a technology analyst that evaluates tech trends and innovations. Provide a concise, insightful analysis.",
    "Economist":             "You are an economist specializing in macro and microeconomic analysis. Provide clear, data-driven insights.",
    "Marketing Strategist":  "You are a marketing strategist focused on positioning and customer insights. Offer strategic recommendations.",
    "Legal Advisor":         "You are a legal advisor who interprets regulations. Provide clear legal guidance.",
    "Healthcare Specialist": "You are a healthcare specialist knowledgeable about medical research. Provide evidence-based perspectives."
}

# ——— Theme toggle state ———
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

toggle_label = "🌙 Dark mode" if not st.session_state.dark_mode else "☀️ Light mode"
if st.sidebar.button(toggle_label):
    st.session_state.dark_mode = not st.session_state.dark_mode

# ——— CSS for light & dark themes ———
LIGHT_CSS = """
<style>
:root { --bg: #ffffff; --fg: #000000; }
.stApp { background-color: var(--bg); color: var(--fg); }
.chat-container {
    background: var(--bg);
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
"""
DARK_CSS = """
<style>
:root { --bg: #0e1117; --fg: #f0f0f0; }
.stApp { background-color: var(--bg); color: var(--fg); }
.chat-container {
    background: var(--bg);
    border: 1px solid #333;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
"""
st.markdown(DARK_CSS if st.session_state.dark_mode else LIGHT_CSS, unsafe_allow_html=True)

# ——— Sidebar: Presets & Parameters ———
st.sidebar.title("⚙️ Settings")

PRESETS = {
    "Custom":            {"system": "You are a helpful assistant.",                  "temperature": 0.7, "max_tokens": 500, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},
    "Legal Assistant":   {"system": "You are a concise legal assistant. Provide clear, structured answers.", "temperature": 0.2, "max_tokens": 400, "top_p": 0.9, "frequency_penalty": 0.0, "presence_penalty": 0.1},
    "Marketing Copywriter":{"system": "You are a creative marketing copywriter. Craft catchy, persuasive copy.", "temperature": 0.9, "max_tokens": 600, "top_p": 1.0, "frequency_penalty": 0.2, "presence_penalty": 0.2},
    "Math Tutor":        {"system": "You are a friendly math tutor. Explain each step in detail.",  "temperature": 0.3, "max_tokens": 500, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},
}

preset_choice = st.sidebar.selectbox("Choose a Preset", list(PRESETS.keys()))
preset        = PRESETS[preset_choice]

# Override system prompt & parameters
system_prompt     = st.sidebar.text_area("System prompt", value=preset["system"], height=100)
temperature       = st.sidebar.slider("Temperature", 0.0, 1.0, value=preset["temperature"])
max_tokens        = st.sidebar.slider("Max tokens", 100, 2000, value=preset["max_tokens"])
top_p             = st.sidebar.slider("Top‑p", 0.0, 1.0, value=preset["top_p"])
frequency_penalty = st.sidebar.slider("Frequency penalty", -2.0, 2.0, value=preset["frequency_penalty"])
presence_penalty  = st.sidebar.slider("Presence penalty", -2.0, 2.0, value=preset["presence_penalty"])
n_completions     = st.sidebar.slider("Number of completions (n)", 1, 5, value=1)
streaming         = st.sidebar.checkbox("Stream responses", value=False)
model             = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])

# Sidebar controls to clear histories
if st.sidebar.button("Clear Chat"):
    st.session_state.history = [{"role": "system", "content": system_prompt}]
if st.sidebar.button("Reset Research"):
    st.session_state.research_session = {"history": [], "summary": ""}

enable_search = st.sidebar.checkbox("Enable web search grounding", value=False)

# ——— Initialize histories ———
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": system_prompt}]
if "research_session" not in st.session_state:
    st.session_state.research_session = {"history": [], "summary": ""}

# ——— Main UI ———
st.title("🧠 Chat & Multi‑Agent Research")
tab_chat, tab_research = st.tabs(["💬 Chat", "🔍 Research"])

# — Chat Tab ——
with tab_chat:
    st.divider()
    # Render chat history
    for msg in st.session_state.history:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])

    # User input
    if user_input := st.chat_input("Your message"):
        # Optional web grounding
        if enable_search:
            web_ctx = google_search(user_input)
            if web_ctx:
                st.session_state.history.insert(
                    1,
                    {"role": "system", "content": "Web context:\n" + "\n".join(web_ctx)}
                )
        # Append and display user message
        st.session_state.history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Call OpenAI
        params = {
            "model": model,
            "messages": st.session_state.history,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n_completions,
            "stream": streaming,
        }
        assistant_msg = ""
        if streaming:
            stream_resp = client.chat.completions.create(**params)
            placeholder = st.chat_message("assistant")
            for chunk in stream_resp:
                delta = chunk.choices[0].delta.get("content", "")
                assistant_msg += delta
                placeholder.write(assistant_msg)
        else:
            resp = client.chat.completions.create(**params)
            assistant_msg = resp.choices[0].message.content.strip()
            st.chat_message("assistant").write(assistant_msg)

        # Save assistant response
        st.session_state.history.append({"role": "assistant", "content": assistant_msg})

# — Research Tab ——
with tab_research:
    st.header("🔍 Multi‑Agent Research")

    # Reset research session
    if st.button("Reset Research Session"):
        st.session_state.research_session = {"history": [], "summary": ""}

    # Show past research entries
    for entry in st.session_state.research_session["history"]:
        st.markdown(f"**Query:** {entry['query']}")
        if entry.get("context"):
            st.write("**Web context:**")
            for c in entry["context"]:
                st.write("-", c)
        for name, out in entry["responses"].items():
            st.markdown(f"**{name}:** {out}")
        st.markdown(f"**Summary:** {entry['summary']}")
        st.divider()

    # Agent selector
    selected_agents = st.multiselect(
        "Select agents to involve",
        options=list(AGENTS.keys()),
        default=list(AGENTS.keys())
    )

    # New research query
    query = st.text_input("Enter research query:", key="research_input")
    if st.button("Run Research"):
        # 1) Web grounding
        context = None
        if enable_search:
            results = google_search(query)
            if results:
                st.write("**Web context:**")
                for r in results:
                    st.write("-", r)
                context = results

        # 2) Call each selected agent, include prior summary
        responses = {}
        prev_sum = st.session_state.research_session["summary"]
        for name in selected_agents:
            prompt = AGENTS[name]
            messages = [{"role": "system", "content": prompt}]
            if prev_sum:
                messages.append({"role": "system", "content": "Previous summary:\n" + prev_sum})
            if context:
                messages.append({"role": "system", "content": "Web context:\n" + "\n".join(context)})
            messages.append({"role": "user", "content": query})

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=1,
                stream=False,
            )
            out = resp.choices[0].message.content.strip()
            st.subheader(name)
            st.write(out)
            responses[name] = out

        # 3) Update consolidated summary
        combined = "\n\n".join(f"{n}:\n{t}" for n, t in responses.items())
        summary_msgs = [
            {"role": "system", "content": "You are a summarizer. Update the prior summary with these new perspectives."},
            {"role": "system", "content": prev_sum},
            {"role": "user", "content": combined}
        ]
        summ = client.chat.completions.create(
            model=model,
            messages=summary_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            stream=False,
        )
        summary = summ.choices[0].message.content.strip()
        st.subheader("Updated Summary")
        st.write(summary)

        # 4) Save session state (Streamlit will rerun automatically)
        st.session_state.research_session["history"].append({
            "query": query,
            "context": context,
            "responses": responses,
            "summary": summary
        })
        st.session_state.research_session["summary"] = summary
