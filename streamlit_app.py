"""
streamlit_app.py - Music Recommendation System Streamlit UI

Reuses MusicRecommendationApp from app.py as the backend (zero backend logic changes).
Run: python -m streamlit run streamlit_app.py
"""

import streamlit as st

from app import MusicRecommendationApp


# ==================== Backend init (cached so the vector index is built once) ====================

@st.cache_resource
def get_app() -> MusicRecommendationApp:
    """Initialize the backend: vector index, LLM engine, CF, user manager."""
    return MusicRecommendationApp()


# ==================== State ====================

def init_state():
    defaults = {
        "user_id": "",
        "username": "",
        "logged_in": False,
        "chat_history": [],
        "feedback_msg": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ==================== Auth ====================

def do_login(app, username, password):
    if not username or not password:
        st.error("Please enter both username and password.")
        return
    success, msg, profile = app.user_manager.login(username, password)
    if success:
        st.session_state.user_id = profile.user_id
        st.session_state.username = profile.username
        st.session_state.logged_in = True
        st.session_state.chat_history = []
        try:
            app.cf_engine.refresh_matrix()
        except Exception:
            pass
        st.rerun()
    else:
        st.error(msg)


def do_register(app, username, password, confirm):
    if password != confirm:
        st.error("Passwords do not match.")
        return
    success, msg, _ = app.user_manager.register(username, password)
    if success:
        st.success(f"{msg} Please switch to the \"Login\" tab.")
    else:
        st.error(msg)


def do_guest(app):
    profile = app.user_manager.guest_login()
    st.session_state.user_id = profile.user_id
    st.session_state.username = profile.username
    st.session_state.logged_in = True
    st.session_state.chat_history = []
    st.rerun()


def do_logout(app):
    if st.session_state.user_id:
        app.user_manager.logout(st.session_state.user_id)
    st.session_state.user_id = ""
    st.session_state.username = ""
    st.session_state.logged_in = False
    st.session_state.chat_history = []
    st.rerun()


# ==================== Sidebar ====================

def render_sidebar(app):
    with st.sidebar:
        st.markdown("## 🎵 Music Recommender")
        st.caption("LLM · Collaborative Filtering · Explainable")

        st.divider()

        if not st.session_state.logged_in:
            tab_login, tab_register = st.tabs(["Login", "Register"])

            with tab_login:
                u = st.text_input("Username", key="login_username")
                p = st.text_input("Password", type="password", key="login_password")
                if st.button("Log in", type="primary", use_container_width=True):
                    do_login(app, u, p)

            with tab_register:
                ru = st.text_input("Username (2-20 chars)", key="reg_username")
                rp = st.text_input("Password (min 4 chars)", type="password", key="reg_password")
                rc = st.text_input("Confirm password", type="password", key="reg_confirm")
                if st.button("Register", use_container_width=True):
                    do_register(app, ru, rp, rc)

            st.divider()
            if st.button("🎭 Continue as guest", use_container_width=True):
                do_guest(app)

        else:
            st.markdown(f"### 👤 {st.session_state.username}")
            if st.button("🚪 Log out", use_container_width=True):
                do_logout(app)

            st.divider()

            with st.expander("👤 User profile", expanded=True):
                st.markdown(app.get_user_profile_summary(st.session_state.user_id))

            with st.expander("📊 Database stats", expanded=False):
                st.markdown(app.get_database_stats())


# ==================== Main area ====================

def inject_css():
    st.markdown(
        """
        <style>
            /* Hero title card */
            .hero {
                background: linear-gradient(120deg, #6d28d9 0%, #8b5cf6 45%, #db2777 100%);
                color: #fff;
                padding: 2rem 2.2rem;
                border-radius: 18px;
                margin-bottom: 1.4rem;
                box-shadow: 0 12px 32px rgba(109, 40, 217, 0.25);
            }
            .hero h1 { color: #fff; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0.3rem; }
            .hero p { color: rgba(255, 255, 255, 0.92); font-size: 1.05rem; margin: 0; }

            /* Rounded buttons */
            .stButton > button { border-radius: 12px; font-weight: 600; }

            /* Rounded chat input */
            [data-testid="stChatInput"] textarea { border-radius: 14px; }

            /* Soft sidebar background */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #faf9ff 0%, #f0eaff 100%);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>🎵 Smart Music Recommender</h1>
            <p>Describe the music you want in natural language · feedback &amp; explainable recommendations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat(app):
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "Tell me what kind of music you'd like… (e.g. \"recommend calm instrumental music for studying\")"
    )

    if prompt:
        if not st.session_state.logged_in:
            st.warning("Please log in or continue as guest first.")
            return

        with st.spinner("Recommending for you…"):
            _, st.session_state.chat_history = app.process_message(
                prompt,
                st.session_state.chat_history,
                st.session_state.user_id,
                st.session_state.username,
            )
        st.rerun()


def render_feedback(app):
    """Like/dislike feedback (only shown when logged in with a recommendation)."""
    if not st.session_state.logged_in:
        return

    profile = app.user_manager.get_user_profile(st.session_state.user_id)
    if not profile or not profile.conversation_history:
        return

    last_ids = profile.conversation_history[-1].recommended_songs or []
    if not last_ids:
        return

    with st.expander("👍 Rate the recommendations (helps the system learn)", expanded=False):
        if st.session_state.get("feedback_msg"):
            st.success(st.session_state.feedback_msg)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("👍 Like all", use_container_width=True):
                st.session_state.feedback_msg = app.provide_feedback(st.session_state.user_id, "like")
                st.rerun()
        with c2:
            if st.button("👎 Dislike all", use_container_width=True):
                st.session_state.feedback_msg = app.provide_feedback(st.session_state.user_id, "dislike")
                st.rerun()

        options = []
        for i, sid in enumerate(last_ids, 1):
            item = app.music_db.get_by_id(sid)
            name = f"{item.song} - {item.artist}" if item else "Unknown song"
            options.append(f"{i}. {name} | {sid}")

        sel = st.selectbox("Rate a single song", options)
        c3, c4 = st.columns(2)
        with c3:
            if st.button("👍 Like this one", use_container_width=True):
                st.session_state.feedback_msg = app.provide_feedback(st.session_state.user_id, "like", sel)
                st.rerun()
        with c4:
            if st.button("👎 Dislike this one", use_container_width=True):
                st.session_state.feedback_msg = app.provide_feedback(st.session_state.user_id, "dislike", sel)
                st.rerun()


# ==================== Entrypoint ====================

def main():
    st.set_page_config(
        page_title="🎵 Smart Music Recommender",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    app = get_app()
    init_state()
    inject_css()

    render_sidebar(app)
    render_hero()

    if st.session_state.logged_in:
        render_chat(app)
        render_feedback(app)
    else:
        st.info("👈 Please log in, register, or continue as guest to start getting recommendations.")


if __name__ == "__main__":
    main()
