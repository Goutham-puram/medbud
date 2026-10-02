"""MedBud chat UI (Streamlit). Talks ONLY to the FastAPI backend over HTTP: login returns a signed
token, every chat call sends it as a Bearer header, and the role shown here is whatever the server
put in the token. Run:  streamlit run ui/app.py   (API at MEDIBOT_API_URL, default localhost:8000)

Session persistence: Streamlit forgets st.session_state on a page reload, so the token is also
kept in a browser cookie (SameSite=strict, same lifetime as the JWT). On reload the app reads the
cookie and asks the API (/me) to validate it before restoring the session. A production client
would use an httpOnly cookie set by the server; a component-set cookie cannot be httpOnly.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import httpx
import streamlit as st
from streamlit_cookies_controller import CookieController

API = os.getenv("MEDIBOT_API_URL", "http://localhost:8000")
COOKIE = "medbud_token"
SESSION_HOURS = int(os.getenv("JWT_HOURS", "8"))
DEMO = [("dr.mehta", "doctor123", "doctor"), ("nurse.priya", "nurse123", "nurse"),
        ("billing.ravi", "billing123", "billing_executive"), ("tech.anand", "tech123", "technician"),
        ("admin.sys", "admin123", "admin")]
LABEL = {"hybrid_rag": "Hybrid RAG", "sql_rag": "SQL RAG", "direct": "No retrieval"}
LABEL_HELP = ("Hybrid RAG: answered from passages of the documents your role may read, with sources. "
              "SQL RAG: computed from the claims and maintenance databases (analytics roles only). "
              "No retrieval: greetings and questions about MedBud itself, answered directly.")
COLLECTION_WORDS = {
    "general": "general staff policies (handbook, leave, code of conduct, FAQs)",
    "clinical": "clinical protocols, drug formulary and diagnostic references",
    "nursing": "nursing procedures and infection control",
    "billing": "insurance billing codes and claim guides",
    "equipment": "equipment operation and maintenance manuals",
}
COLLECTION_OWNERS = {
    "clinical": "doctors and administrators", "nursing": "nurses, doctors and administrators",
    "billing": "billing executives and administrators", "equipment": "technicians and administrators",
    "general": "everyone",
}
STARTERS = {
    "doctor": ["What is the first-line therapy for community-acquired pneumonia?",
               "Which drugs need renal dose adjustment?", "What are the danger signs in paediatric fever?",
               "Which equipment category has the most open maintenance tickets?"],
    "nurse": ["What is the correct IV cannula size for a paediatric patient?",
              "What are the five moments of hand hygiene?", "How many casual leaves do I get per year?",
              "Show me the insurance billing codes"],
    "billing_executive": ["Which documents are needed for pre-authorisation?",
                          "How many billing claims were escalated last month?",
                          "What is the total approved amount per insurer?", "What is the metformin dose for diabetes?"],
    "technician": ["What does fault code E-07 mean on the infusion pump?",
                   "What is the preventive maintenance schedule for the X-ray unit?",
                   "What is the dress code?", "How many maintenance tickets are still open?"],
    "admin": ["Which equipment category has the most open maintenance tickets?",
              "What is the notice period for resignation?", "What PPE is required for airborne precautions?",
              "What is the room rent sub-limit for Star Health?"],
}

st.set_page_config(page_title="MedBud", layout="wide")
cookies = CookieController()

# Cookie writes are component calls that only reach the browser when the script run completes,
# so they are never followed by st.rerun() in the same run: login/logout queue the write, rerun,
# and the write happens at the top of the next run.
if "cookie_to_set" in st.session_state:
    cookies.set(COOKIE, st.session_state.pop("cookie_to_set"), max_age=SESSION_HOURS * 3600,
                expires=datetime.now() + timedelta(hours=SESSION_HOURS))
if st.session_state.pop("cookie_to_clear", False):
    cookies.remove(COOKIE)


def api_post(path: str, json: dict, token: str | None = None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.post(f"{API}{path}", json=json, headers=headers, timeout=90)


def api_get(path: str, token: str | None = None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.get(f"{API}{path}", headers=headers, timeout=30)


def start_session(token: str, user: dict) -> None:
    st.session_state.token = token
    st.session_state.user = user
    st.session_state.messages = []


def logout() -> None:
    for key in ("token", "user", "messages"):
        st.session_state.pop(key, None)
    st.session_state.cookie_to_clear = True


def restore_from_cookie() -> bool:
    """After a reload: validate the cookie's token with the API before trusting it."""
    token = cookies.get(COOKIE)
    if not token:
        return False
    try:
        r = api_get("/me", token=token)
    except httpx.HTTPError:
        return False
    if r.status_code != 200:
        cookies.remove(COOKIE)
        return False
    start_session(token, r.json())
    return True


def login_screen() -> None:
    st.title("MedBud")
    st.caption("MediAssist Health Network internal assistant. Sign in with one of the demo accounts.")
    left, right = st.columns([1, 1])
    with left:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in"):
                try:
                    r = api_post("/login", {"username": username, "password": password})
                except httpx.HTTPError as exc:
                    st.error(f"Cannot reach the API at {API}: {exc}")
                    return
                if r.status_code != 200:
                    st.error(r.json().get("detail", "login failed"))
                    return
                data = r.json()
                start_session(data["access_token"], data)
                st.session_state.cookie_to_set = data["access_token"]
                st.rerun()
    with right:
        st.markdown("**Demo accounts**")
        st.table([{"username": u, "password": p, "role": r} for u, p, r in DEMO])


def sidebar(user: dict) -> None:
    with st.sidebar:
        st.markdown(f"### {user['username']}")
        st.badge(user["role"].replace("_", " "), help="Your role, as verified by the server at sign-in. It decides which documents you can read.")
        st.caption("Can read: " + ", ".join(f"`{c}`" for c in user["collections"]),
                   help="Document collections your role may search. " +
                        " ".join(f"{c}: {COLLECTION_WORDS[c]}." for c in user["collections"]))
        st.caption("Analytics (SQL RAG): " + ("yes" if user["sql_access"] else "no"),
                   help="Numbers questions (counts, totals, trends) are answered from the claims and maintenance "
                        "databases. Available to billing executives and administrators.")
        st.divider()
        if st.button("Sign out"):
            logout()
            st.rerun()


def readable(collections: list[str]) -> str:
    names = [c for c in collections]
    return ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]


def welcome_text(user: dict) -> str:
    cols = readable(user["collections"])
    hidden = [c for c in COLLECTION_WORDS if c not in user["collections"]]
    scope = f"I answer from MediAssist documents your role can read: **{cols}**."
    if user["sql_access"]:
        scope += " I can also answer **numbers questions** (counts, totals, trends) from the claims and maintenance databases."
    limits = (f" I can't see {readable(hidden)} material," if hidden else " I can see every collection,") + \
             " and I don't replace clinical or professional judgement."
    return scope + limits + " Try one of the questions below, or ask your own."


def help_popover(user: dict) -> None:
    with st.popover("?", help="How MedBud works"):
        st.markdown("#### What MedBud is")
        st.markdown("An assistant that answers staff questions from MediAssist's internal documents and, "
                    "for analytics roles, from the claims and maintenance databases. It only ever reads "
                    "material your role is allowed to see.")
        st.markdown("#### What you can ask")
        st.markdown("\n".join(f"- **{c}** — {COLLECTION_WORDS[c]}" for c in user["collections"]))
        if user["sql_access"]:
            st.markdown("- **numbers** — e.g. *how many claims were rejected last month?* (computed from the database)")
        st.markdown("#### How an answer is made")
        st.markdown("1. MedBud finds the passages that match your question — only among documents your role may read.\n"
                    "2. It ranks them and keeps the best three.\n"
                    "3. It writes a short answer from those passages and lists them as **Sources** under the reply.")
        st.markdown("#### Why was I refused?")
        st.markdown("If a question is about documents outside your role, or nothing relevant exists in your "
                    "documents, MedBud says so instead of guessing. Access is enforced when the documents are "
                    "searched, so restricted material is never read on your behalf.")
        st.markdown("#### Tips")
        st.markdown("- Name the drug, device model, code or policy you mean.\n- One question at a time.\n"
                    "- Check the **Sources** before acting on an answer.")


def why_refused(user: dict) -> None:
    with st.expander("Why?"):
        st.markdown(f"Your role (**{user['role'].replace('_', ' ')}**) can read **{readable(user['collections'])}** documents."
                    + (" You can also run database analytics." if user["sql_access"] else
                       " Database analytics are available to billing executives and administrators."))
        hidden = [c for c in COLLECTION_WORDS if c not in user["collections"]]
        if hidden:
            st.markdown("Restricted for you: " + "; ".join(f"**{c}** ({COLLECTION_OWNERS[c]})" for c in hidden) + ".")
        st.markdown("If you need that information, ask a colleague in that department or your supervisor.")


def render_assistant(msg: dict) -> None:
    label = f"{LABEL.get(msg['retrieval_type'], msg['retrieval_type'])} · {msg['latency_ms']} ms"
    if msg.get("denied"):
        st.warning(msg["answer"])
        if "user" in st.session_state:
            why_refused(st.session_state.user)
    else:
        st.markdown(msg["answer"])
    st.caption(label, help=LABEL_HELP)
    if msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for s in msg["sources"]:
                st.markdown(f"- **{s['source_document']}** — {s['section_title']} (`{s['collection']}`)")
    if msg.get("sql"):
        with st.expander("SQL used"):
            st.code(msg["sql"], language="sql")


def chat_screen() -> None:
    user = st.session_state.user
    sidebar(user)
    title_col, help_col = st.columns([12, 1])
    with title_col:
        st.title("MedBud")
    with help_col:
        st.write("")
        help_popover(user)
    with st.chat_message("assistant"):  # scope + limits stay visible as the first message
        st.markdown(welcome_text(user))
    for m in st.session_state.messages:
        with st.chat_message(m["who"]):
            if m["who"] == "user":
                st.markdown(m["text"])
            else:
                render_assistant(m)

    picked = None
    if not st.session_state.messages:  # empty state: concrete starter questions (click to ask)
        picked = st.pills("Try asking", STARTERS[user["role"]], selection_mode="single", key="starter",
                          label_visibility="collapsed", help="Click a question to ask it")
    question = st.chat_input("Ask about protocols, policies, equipment, billing - or analytics if your role allows")
    question = question or picked
    if not question:
        return
    st.session_state.messages.append({"who": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = api_post("/chat", {"question": question}, token=st.session_state.token)
            except httpx.HTTPError as exc:
                st.error(f"API error: {exc}")
                return
        if r.status_code == 401:
            st.error("Session expired - please sign in again.")
            logout()
            st.rerun()
        if r.status_code != 200:
            st.error(r.json().get("detail", f"request failed ({r.status_code})"))
            return
        data = r.json()
        data["who"] = "assistant"
        st.session_state.messages.append(data)
        render_assistant(data)
    st.rerun()  # redraw from history: starters disappear after the first question


# First run of a fresh browser session: the cookie component has not reported yet, so wait one
# run before deciding between "restore" and "show login" (avoids a login-form flash on reload).
if "token" not in st.session_state and not st.session_state.get("first_run_done"):
    st.session_state.first_run_done = True
    if "cookies" not in st.session_state:
        with st.spinner("Loading..."):
            st.stop()

if "token" not in st.session_state:
    restore_from_cookie()

if "token" in st.session_state:
    chat_screen()
else:
    login_screen()
