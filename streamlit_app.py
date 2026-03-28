"""
streamlit_app.py  —  Entropic ASR Demo Interface
-------------------------------------------------
Run:  streamlit run streamlit_app.py --server.port 8501

pip install streamlit>=1.31.0 requests

Three problems fixed vs previous version:
  1. Audio recording  — uses st.audio_input() (Streamlit 1.31+, cleaner than
                        streamlit-mic-recorder). Still requires browser mic
                        permission. On remote DGX over plain HTTP, do this once:
                        chrome://flags → "Insecure origins treated as secure"
                        → add http://<dgx-ip>:8501 → relaunch Chrome.
  2. State on refresh — session_id stored in URL query param (?sid=...).
                        All display state written to /tmp/entropic_ui/<sid>.json
                        after every turn and restored on page load.
  3. New file ignored — audio bytes are MD5-hashed. API is called only when the
                        hash changes, not on every Streamlit rerun.
"""

import json
import time
import hashlib
import os
import requests
import tempfile
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Entropic ASR",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0d0d;
    color: #e0e0e0;
  }
  section[data-testid="stSidebar"] { background: #111; border-right: 1px solid #2a2a2a; }
  h1,h2,h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
  h1 { color: #f5f5f5; font-size: 1.4rem; font-weight: 600; }
  h3 { color: #aaa; font-size: 0.85rem; font-weight: 400; text-transform: uppercase; letter-spacing: 2px; }

  .card         { background:#161616; border:1px solid #2a2a2a; border-radius:4px; padding:1.2rem 1.4rem; margin-bottom:0.8rem; }
  .card-accent  { border-left:3px solid #f0a500; }

  .badge        { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                  padding:2px 8px; border-radius:2px; font-weight:600; letter-spacing:1px; }
  .badge-accept { background:#0d3321; color:#3ddc84; border:1px solid #3ddc84; }
  .badge-soft   { background:#2d2000; color:#f0a500; border:1px solid #f0a500; }
  .badge-hard   { background:#2d0a0a; color:#ff4444; border:1px solid #ff4444; }
  .badge-kw     { background:#0d1f33; color:#64b5f6; border:1px solid #64b5f6; }

  .lat-row      { display:flex; align-items:center; gap:8px; margin:3px 0;
                  font-family:'IBM Plex Mono',monospace; font-size:0.72rem; }
  .lat-label    { width:90px; color:#888; }
  .lat-bar-bg   { flex:1; background:#222; border-radius:2px; height:8px; overflow:hidden; }
  .lat-bar-fill { height:100%; border-radius:2px; background:#f0a500; }
  .lat-val      { width:55px; text-align:right; color:#ccc; }

  .transcript-box { background:#0a0a0a; border:1px solid #1f1f1f; border-radius:3px;
                    padding:1rem; font-family:'IBM Plex Mono',monospace; font-size:0.85rem;
                    color:#c8e6c9; min-height:48px; }
  .agent-box    { background:#0f1a2e; border:1px solid #1a3a6a; border-radius:3px;
                  padding:1rem; font-size:0.95rem; color:#90caf9; min-height:48px; }

  .slot-table   { width:100%; border-collapse:collapse; font-size:0.82rem; }
  .slot-table td { padding:6px 10px; border-bottom:1px solid #1a1a1a; font-family:'IBM Plex Mono',monospace; }
  .slot-table td:first-child { color:#888; width:40%; }
  .slot-table td:last-child  { color:#fff; }

  .final-box    { background:#0a1a0a; border:1px solid #1a4a1a; border-radius:4px;
                  padding:1.2rem; font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
                  color:#a5d6a7; white-space:pre-wrap; }

  .metric-grid  { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:8px; }
  .metric-cell  { background:#111; border:1px solid #222; border-radius:3px; padding:10px; text-align:center; }
  .metric-val   { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; color:#f0a500; }
  .metric-lbl   { font-size:0.68rem; color:#666; text-transform:uppercase; letter-spacing:1px; margin-top:2px; }

  .turn-entry   { border-left:2px solid #2a2a2a; padding:6px 0 6px 12px; margin:4px 0;
                  font-size:0.78rem; font-family:'IBM Plex Mono',monospace; color:#888; }
  .turn-entry .tier-accept { color:#3ddc84; }
  .turn-entry .tier-soft   { color:#f0a500; }
  .turn-entry .tier-hard   { color:#ff4444; }
  .turn-entry .tier-kw     { color:#64b5f6; }

  .topbar       { display:flex; align-items:center; gap:16px; padding:10px 0 18px 0;
                  border-bottom:1px solid #1f1f1f; margin-bottom:20px; }
  .topbar-title { font-family:'IBM Plex Mono',monospace; font-size:1.2rem; font-weight:600;
                  color:#f5f5f5; letter-spacing:-0.5px; }
  .topbar-sub   { font-size:0.78rem; color:#555; letter-spacing:1px; text-transform:uppercase; }

  .mic-notice   { background:#1a1500; border:1px solid #3a2e00; border-radius:4px;
                  padding:0.7rem 1rem; font-size:0.75rem; color:#c8a800;
                  font-family:'IBM Plex Mono',monospace; line-height:1.7; margin-bottom:1rem; }

  .stButton>button { background:#f0a500 !important; color:#000 !important; border:none !important;
    border-radius:2px !important; font-family:'IBM Plex Mono',monospace !important;
    font-weight:600 !important; letter-spacing:1px !important; text-transform:uppercase !important;
    font-size:0.78rem !important; padding:8px 20px !important; }
  .stButton>button:hover { background:#d49200 !important; }

  div[data-testid="stSelectbox"] label,
  div[data-testid="stTextInput"] label { color:#888 !important; font-size:0.78rem !important; }
  .stSelectbox>div>div  { background:#111 !important; border-color:#2a2a2a !important; color:#e0e0e0 !important; }
  .stTextInput>div>div>input { background:#111 !important; border-color:#2a2a2a !important; color:#e0e0e0 !important; }
  div[data-testid="stFileUploader"] { background:#111 !important; border:1px dashed #333 !important; border-radius:4px !important; }
  div[data-testid="stAudioInput"]   { background:#111 !important; border:1px solid #2a2a2a !important; border-radius:4px !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE  = "http://localhost:8000"
CACHE_DIR = "/tmp/entropic_ui"
os.makedirs(CACHE_DIR, exist_ok=True)

USE_CASES = {
    "🚔  FIR / Police Statement": {
        "description": "Record a victim or witness statement directly in Hinglish. "
                       "The system extracts a structured FIR-ready record.",
        "session_prefix": "fir",
        "example": "kal raat mere dukan mein ghuse, teen log the, paanch hazaar le gaye",
    },
    "🏡  Oral Asset Declaration": {
        "description": "Elderly rural property owners speak their asset distribution wishes. "
                       "System produces a documented record.",
        "session_prefix": "asset",
        "example": "meri zameen do bigha gaon mein, bade bete ko deni hai",
    },
    "🏥  ASHA Health Record": {
        "description": "Community health workers dictate observations from home visits. "
                       "System generates a structured child health record.",
        "session_prefix": "health",
        "example": "teen saal ka baccha, weight barah kilo, khana nahi khata",
    },
    "💸  Financial Transaction": {
        "description": "Voice-driven UPI / payment commands in Hinglish.",
        "session_prefix": "finance",
        "example": "das hazaar rupaye Rahul ko bhejo",
    },
}

TIER_BADGE = {
    "ACCEPT":        '<span class="badge badge-accept">ACCEPT</span>',
    "ACCEPT_KW":     '<span class="badge badge-accept">ACCEPT</span> <span class="badge badge-kw">KW</span>',
    "SOFT_REPROMPT": '<span class="badge badge-soft">SOFT</span>',
    "HARD_REPROMPT": '<span class="badge badge-hard">HARD</span>',
}

# Keys that are saved to disk and restored on refresh
PERSISTENT_KEYS = [
    "session_id", "turn_log", "agent_prompt", "collected_slots",
    "missing_slots", "final_record", "eval_summary",
    "last_transcript", "last_normalized", "last_conf", "last_raw_conf",
    "last_tier", "last_kw_override", "last_latency", "conversation_done",
    "last_audio_hash",
]


# ── Disk-backed session cache ─────────────────────────────────────────────────
def _cache_path(sid: str) -> str:
    return os.path.join(CACHE_DIR, f"{sid}.json")

def save_ui_state(sid: str):
    """Write all persistent session state keys to disk."""
    if not sid:
        return
    snapshot = {}
    for k in PERSISTENT_KEYS:
        v = st.session_state.get(k)
        if v is not None:
            snapshot[k] = v
    try:
        with open(_cache_path(sid), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, default=str)
    except Exception:
        pass  # non-fatal; UI still works, just won't survive refresh

def load_ui_state(sid: str) -> bool:
    """Restore state from disk. Returns True if cache was found."""
    path = _cache_path(sid)
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            st.session_state[k] = v
        return True
    except Exception:
        return False

def delete_ui_state(sid: str):
    try:
        os.remove(_cache_path(sid))
    except Exception:
        pass


# ── Session state defaults ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "session_id":        None,
        "turn_log":          [],
        "agent_prompt":      "",
        "collected_slots":   {},
        "missing_slots":     [],
        "final_record":      None,
        "eval_summary":      None,
        "last_transcript":   "",
        "last_normalized":   "",
        "last_conf":         None,
        "last_raw_conf":     None,
        "last_tier":         None,
        "last_kw_override":  False,
        "last_latency":      {},
        "conversation_done": False,
        "last_audio_hash":   None,  # MD5 of last processed audio — prevents re-processing on rerun
        "_state_restored":   False, # sentinel: have we tried to restore from disk this session?
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Restore from URL query param on first load ────────────────────────────────
# If the user refreshes the page, ?sid=xxx is still in the URL.
# We load the cached state so the turn log and transcript reappear.
if not st.session_state._state_restored:
    st.session_state._state_restored = True
    params = st.query_params
    if "sid" in params:
        sid = params["sid"]
        if load_ui_state(sid):
            # State restored — session_id already in state from the loaded snapshot
            pass


def reset_session():
    sid = st.session_state.get("session_id")
    if sid:
        delete_ui_state(sid)
    # Clear query param
    try:
        st.query_params.clear()
    except Exception:
        pass
    for k in PERSISTENT_KEYS + ["_state_restored"]:
        if k in st.session_state:
            del st.session_state[k]
    init_state()
    st.rerun()


# ── Audio hash helper ─────────────────────────────────────────────────────────
def audio_md5(audio_bytes: bytes) -> str:
    return hashlib.md5(audio_bytes).hexdigest()


# ── Render helpers ────────────────────────────────────────────────────────────
def render_latency(latency: dict):
    if not latency:
        return
    total  = latency.get("total_ms", 1) or 1
    stages = [("VAD", "vad_ms"), ("ASR", "asr_ms"), ("ITN", "itn_ms"), ("Intent", "intent_ms")]
    html   = ""
    for label, key in stages:
        ms  = latency.get(key, 0) or 0
        pct = min(100, round(ms / total * 100))
        html += (
            f'<div class="lat-row">'
            f'<span class="lat-label">{label}</span>'
            f'<div class="lat-bar-bg"><div class="lat-bar-fill" style="width:{pct}%"></div></div>'
            f'<span class="lat-val">{ms} ms</span>'
            f'</div>'
        )
    html += (
        f'<div class="lat-row" style="margin-top:6px;border-top:1px solid #222;padding-top:4px;">'
        f'<span class="lat-label" style="color:#ccc;font-weight:600;">TOTAL</span>'
        f'<div class="lat-bar-bg"><div class="lat-bar-fill" style="width:100%;background:#fff;"></div></div>'
        f'<span class="lat-val" style="color:#fff;">{total} ms</span>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_slots(collected: dict, missing: list):
    if not collected and not missing:
        return
    rows = "".join(f"<tr><td>✓ {k}</td><td>{v}</td></tr>" for k, v in collected.items())
    rows += "".join(f"<tr><td style='color:#666'>○ {k}</td><td style='color:#444'>—</td></tr>" for k in missing)
    st.markdown(f'<table class="slot-table"><tbody>{rows}</tbody></table>', unsafe_allow_html=True)


def render_eval(ev: dict):
    if not ev:
        return
    total       = max(ev.get("total_turns", 1), 1)
    accept_rate = round(ev.get("accepts", 0) / total * 100)
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-cell"><div class="metric-val">{ev.get('total_turns',0)}</div><div class="metric-lbl">Total Turns</div></div>
      <div class="metric-cell"><div class="metric-val" style="color:#3ddc84">{ev.get('accepts',0)}</div><div class="metric-lbl">Accepted</div></div>
      <div class="metric-cell"><div class="metric-val" style="color:#f0a500">{ev.get('soft_reprompts',0)}</div><div class="metric-lbl">Soft Reprompts</div></div>
      <div class="metric-cell"><div class="metric-val" style="color:#ff4444">{ev.get('hard_reprompts',0)}</div><div class="metric-lbl">Hard Reprompts</div></div>
      <div class="metric-cell"><div class="metric-val">{ev.get('avg_confidence',0):.2f}</div><div class="metric-lbl">Avg Confidence</div></div>
      <div class="metric-cell"><div class="metric-val" style="color:#3ddc84">{accept_rate}%</div><div class="metric-lbl">Accept Rate</div></div>
    </div>""", unsafe_allow_html=True)


# ── API call ──────────────────────────────────────────────────────────────────
def send_audio(audio_bytes: bytes, session_id: str) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        with open(tmp, "rb") as f:
            resp = requests.post(
                f"{API_BASE}/chat",
                data={"session_id": session_id},
                files={"audio": ("audio.wav", f, "audio/wav")},
                timeout=30,
            )
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None
    finally:
        os.unlink(tmp)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ENTROPIC ASR")
    st.markdown(
        '<div style="color:#555;font-size:0.72rem;letter-spacing:2px;'
        'text-transform:uppercase;margin-bottom:20px">Privacy-First Voice Intelligence</div>',
        unsafe_allow_html=True,
    )

    use_case_name = st.selectbox("Use Case", list(USE_CASES.keys()), key="use_case_select")
    uc = USE_CASES[use_case_name]

    st.markdown(
        f'<div class="card" style="margin-top:8px">'
        f'<div style="font-size:0.78rem;color:#888;line-height:1.5">{uc["description"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.session_state.session_id:
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
            f'color:#555;margin-bottom:8px">SESSION<br>'
            f'<span style="color:#888">{st.session_state.session_id}</span></div>',
            unsafe_allow_html=True,
        )

    if st.button("NEW SESSION"):
        reset_session()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.68rem;color:#333;line-height:1.6">'
        'Audio runs locally on your server.<br>'
        'No data leaves the institution.<br>'
        'Whisper + DistilBERT + Qwen 0.5B.</div>',
        unsafe_allow_html=True,
    )


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="topbar">'
    f'<div><div class="topbar-title">🎙️ ENTROPIC ASR</div>'
    f'<div class="topbar-sub">{use_case_name.split("  ")[-1]}</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([3, 2], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("### VOICE INPUT")

    # ── Mic setup notice (shown when no recording has happened yet) ───────
    # Explains the one-time Chrome flag step for remote DGX.
    # Dismissed after first successful turn.
    if not st.session_state.last_transcript:
        st.markdown(
            '<div class="mic-notice">'
            '🔒 <strong>Remote server mic setup (one-time)</strong><br>'
            'Chrome blocks mic on plain HTTP. To enable recording:<br>'
            '1. Go to <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code><br>'
            '2. Add <code>http://&lt;dgx-ip&gt;:8501</code> → Enable → Relaunch Chrome<br>'
            'File upload works immediately without any setup.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Audio input ───────────────────────────────────────────────────────
    # st.audio_input() — available in Streamlit >= 1.31.0
    # Uses the browser's native audio recording widget. Cleaner than
    # streamlit-mic-recorder and does not require a separate package.
    # Falls back to file uploader if the Streamlit version is too old.

    audio_bytes  = None
    input_source = None   # "mic" | "upload"

    has_audio_input = hasattr(st, "audio_input")

    if has_audio_input:
        tab_mic, tab_file = st.tabs(["🎤  Record", "📁  Upload .wav"])

        with tab_mic:
            st.caption("Press the mic button to start/stop recording")
            recorded = st.audio_input(
                "Record audio",
                label_visibility="collapsed",
                key="mic_widget",
            )
            if recorded is not None:
                audio_bytes  = recorded.read()
                input_source = "mic"

        with tab_file:
            uploaded = st.file_uploader(
                "Upload audio",
                type=["wav", "mp3", "m4a", "ogg"],
                label_visibility="collapsed",
                key="file_widget",
            )
            if uploaded is not None:
                audio_bytes  = uploaded.read()
                input_source = "upload"

    else:
        # Streamlit < 1.31 fallback
        st.caption("Upgrade to Streamlit >= 1.31 for mic recording. File upload available now:")
        uploaded = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "m4a", "ogg"],
            label_visibility="collapsed",
            key="file_widget",
        )
        if uploaded is not None:
            audio_bytes  = uploaded.read()
            input_source = "upload"

    st.caption(f'💡 Example: *"{uc["example"]}"*')

    # ── Process audio — only when hash changes ────────────────────────────
    #
    # Root cause of "keeps showing same result":
    # Streamlit reruns the entire script on every widget interaction
    # (tab switch, button click, etc.). Without a hash check, `audio_bytes`
    # is re-sent to the API on every rerun even if nothing new was recorded.
    #
    # Fix: MD5-hash the audio bytes. Only call the API when the hash is
    # different from the last one we processed. After processing, store the
    # new hash so the next rerun does nothing.

    if audio_bytes and not st.session_state.conversation_done:
        current_hash = audio_md5(audio_bytes)

        if current_hash != st.session_state.last_audio_hash:
            # This is genuinely new audio — create session if needed
            if st.session_state.session_id is None:
                prefix = uc["session_prefix"]
                sid    = f"{prefix}_{int(time.time())}"
                st.session_state.session_id = sid
                # Write session_id into URL so refresh restores it
                st.query_params["sid"] = sid

            with st.spinner("Processing audio…"):
                result = send_audio(audio_bytes, st.session_state.session_id)

            # Store hash immediately — even if API failed, don't retry same audio
            st.session_state.last_audio_hash = current_hash

            if result:
                ps = result.get("pipeline_state", result)
                ag = result.get("agent_state", {})

                tier     = ps.get("status", "ACCEPT")
                conf     = ps.get("confidence", 0.0)
                raw_conf = ps.get("raw_confidence", conf)
                kw_flag  = ps.get("keyword_override", False)

                st.session_state.last_transcript  = ps.get("transcript", "")
                st.session_state.last_normalized  = ps.get("normalized_text", "")
                st.session_state.last_conf        = conf
                st.session_state.last_raw_conf    = raw_conf
                st.session_state.last_tier        = tier
                st.session_state.last_kw_override = kw_flag
                st.session_state.last_latency     = ps.get("latency", result.get("latency", {}))

                a_status = ag.get("status", "")
                st.session_state.agent_prompt    = ag.get("agent_prompt", ag.get("message", ""))
                st.session_state.collected_slots = ag.get("collected_slots", ag.get("final_record", {}))
                st.session_state.missing_slots   = ag.get("missing_slots", [])
                st.session_state.eval_summary    = ag.get("eval_summary", ag.get("eval", {}))

                st.session_state.turn_log.append({
                    "turn":       len(st.session_state.turn_log) + 1,
                    "transcript": st.session_state.last_transcript,
                    "tier":       tier,
                    "conf":       conf,
                    "kw":         kw_flag,
                    "source":     input_source,
                })

                if a_status == "complete":
                    st.session_state.final_record      = ag.get("final_record", {})
                    st.session_state.eval_summary      = ag.get("eval_summary", {})
                    st.session_state.conversation_done = True

            # Persist state to disk after every turn so refresh restores it
            save_ui_state(st.session_state.session_id)

        # else: same audio as last turn — do nothing, just render existing state

    # ── Transcript display ─────────────────────────────────────────────────
    if st.session_state.last_transcript:
        st.markdown("### TRANSCRIPT")

        tier     = st.session_state.last_tier or "ACCEPT"
        kw_flag  = st.session_state.last_kw_override
        conf_val = st.session_state.last_conf or 0
        raw_conf = st.session_state.last_raw_conf

        badge_key  = "ACCEPT_KW" if (tier == "ACCEPT" and kw_flag) else tier
        tier_badge = TIER_BADGE.get(badge_key, "")

        conf_display = f"conf={conf_val:.2f}"
        if kw_flag and raw_conf is not None and raw_conf != conf_val:
            conf_display += (
                f' <span style="color:#444;font-size:0.65rem">'
                f'(raw={raw_conf:.2f})</span>'
            )

        st.markdown(
            f'<div class="card card-accent">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            f'<span style="font-size:0.72rem;color:#666;font-family:\'IBM Plex Mono\',monospace">RAW</span>'
            f'<span>{tier_badge} '
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:#888">'
            f'{conf_display}</span></span>'
            f'</div>'
            f'<div class="transcript-box">{st.session_state.last_transcript}</div>'
            f'<div style="font-size:0.7rem;color:#555;margin-top:8px;'
            f'font-family:\'IBM Plex Mono\',monospace">'
            f'ITN → {st.session_state.last_normalized}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Agent prompt ───────────────────────────────────────────────────────
    if st.session_state.agent_prompt and not st.session_state.conversation_done:
        st.markdown("### AGENT")
        st.markdown(
            f'<div class="agent-box">🎙️ &nbsp;{st.session_state.agent_prompt}</div>',
            unsafe_allow_html=True,
        )

    # ── Completed conversation — show final record ─────────────────────────
    if st.session_state.final_record:
        st.markdown("### ✅ STRUCTURED RECORD")
        record_display = {
            k: v for k, v in st.session_state.final_record.items()
            if k != "verbatim"
        }
        st.markdown(
            f'<div class="final-box">'
            f'{json.dumps(record_display, indent=2, ensure_ascii=False)}'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Verbatim transcript accordion
        with st.expander("📜 Verbatim turn log"):
            for line in st.session_state.final_record.get("verbatim", []):
                st.text(line)

        if st.button("📋  COPY JSON"):
            st.code(json.dumps(st.session_state.final_record, indent=2, ensure_ascii=False))

        st.markdown(
            '<div style="font-size:0.75rem;color:#555;margin-top:4px">'
            'Start a new session from the sidebar to record another statement.</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
with col_right:

    if st.session_state.last_latency:
        st.markdown("### LATENCY")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_latency(st.session_state.last_latency)
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.collected_slots or st.session_state.missing_slots:
        st.markdown("### SLOTS")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_slots(
            {k: v for k, v in st.session_state.collected_slots.items()
             if k not in ("verbatim", "total_turns", "intent")},
            st.session_state.missing_slots,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.eval_summary:
        st.markdown("### EVAL METRICS")
        render_eval(st.session_state.eval_summary)

    if st.session_state.turn_log:
        st.markdown("### TURN LOG")
        log_html = ""
        for t in reversed(st.session_state.turn_log):
            tier = t["tier"]
            kw   = t.get("kw", False)
            src  = t.get("source", "")
            src_icon = "🎤" if src == "mic" else "📁" if src == "upload" else ""

            if tier == "ACCEPT" and kw:
                color_class, label = "tier-kw", "ACCEPT+KW"
            else:
                color_class = {
                    "ACCEPT":        "tier-accept",
                    "SOFT_REPROMPT": "tier-soft",
                    "HARD_REPROMPT": "tier-hard",
                }.get(tier, "tier-accept")
                label = tier

            snippet = t["transcript"][:80] + ("…" if len(t["transcript"]) > 80 else "")
            log_html += (
                f'<div class="turn-entry">'
                f'<span class="{color_class}">T{t["turn"]} [{label}]</span>'
                f' <span style="color:#555">conf={t["conf"]:.2f}</span>'
                f' <span style="color:#444">{src_icon}</span><br>'
                f'{snippet}'
                f'</div>'
            )
        st.markdown(log_html, unsafe_allow_html=True)