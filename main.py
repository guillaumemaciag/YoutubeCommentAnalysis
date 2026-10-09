from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from sentiment_analyzer.analysis import analyze_comments, top_keywords
from sentiment_analyzer.chat import RetrievedComment, answer_question
from sentiment_analyzer.youtube import (
    YouTubeAPIError,
    extract_video_id,
    fetch_comments,
    fetch_video_info,
)


st.set_page_config(
    page_title="Comment Pulse",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {"Positive": "#31D0AA", "Neutral": "#8793A6", "Negative": "#FF647C"}


def _secret(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except (FileNotFoundError, KeyError):
        return ""


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
        :root { --ink:#F7F8FC; --muted:#A8B0C2; --panel:#151923; --line:#2A3140; --red:#FF4B62; }
        .stApp { background: radial-gradient(circle at 78% 3%, #242033 0, #0E1117 27rem); color:var(--ink); }
        html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
        h1, h2, h3 { font-family:'Manrope', sans-serif !important; letter-spacing:-0.035em; }
        h1 { font-size:clamp(2.4rem, 6vw, 5rem) !important; line-height:.98 !important; margin-bottom:.6rem !important; }
        [data-testid="stSidebar"] { background:#10131A; border-right:1px solid var(--line); }
        [data-testid="stMetric"] { background:linear-gradient(145deg, #181D28, #12161F); border:1px solid var(--line); padding:1rem 1.1rem; border-radius:16px; }
        [data-testid="stMetricValue"] { font-family:'Manrope', sans-serif; font-size:1.75rem; }
        .hero-kicker { color:#FF7486; font-weight:700; letter-spacing:.16em; text-transform:uppercase; font-size:.76rem; }
        .hero-copy { color:var(--muted); font-size:1.08rem; max-width:720px; margin-bottom:1.8rem; }
        .video-card { padding:1.25rem 1.35rem; border:1px solid var(--line); border-radius:18px; background:rgba(21,25,35,.86); }
        .video-card p { color:var(--muted); margin:.25rem 0 0; }
        .eyebrow { color:#FF7486; font-size:.73rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
        .empty-state { text-align:center; padding:4rem 1.5rem; border:1px dashed #353D4F; border-radius:22px; background:rgba(21,25,35,.55); }
        .empty-state .play { width:64px; height:64px; line-height:64px; border-radius:50%; background:#FF4B62; margin:0 auto 1rem; font-size:1.5rem; }
        .empty-state p { color:var(--muted); max-width:560px; margin:.4rem auto; }
        .stButton button, .stDownloadButton button { border-radius:12px; font-weight:700; }
        .stTabs [data-baseweb="tab-list"] { gap:1.8rem; border-bottom:1px solid var(--line); }
        .stTabs [data-baseweb="tab"] { padding:.85rem .15rem; }
        div[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:16px; background:rgba(21,25,35,.7); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_state() -> None:
    defaults = {"comments": None, "video": None, "messages": [], "chat_sources": {}}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _pretty_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def _overall_sentiment(frame: pd.DataFrame) -> tuple[str, float]:
    score = float(frame["compound"].mean()) if not frame.empty else 0.0
    label = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
    return label, score


def _video_header(video: dict[str, object]) -> None:
    image, details = st.columns([1, 3], vertical_alignment="center")
    with image:
        if video.get("thumbnail"):
            st.image(str(video["thumbnail"]), width="stretch")
    with details:
        st.markdown('<span class="eyebrow">Now analyzing</span>', unsafe_allow_html=True)
        st.subheader(str(video["title"]))
        st.caption(
            f"{video['channel']}  ·  {_pretty_number(int(video['views']))} views  ·  "
            f"{_pretty_number(int(video['likes']))} video likes"
        )
        st.link_button("Open on YouTube ↗", f"https://www.youtube.com/watch?v={video['video_id']}")


def _overview(frame: pd.DataFrame) -> None:
    counts = frame["sentiment"].value_counts()
    total = len(frame)
    overall, mean_score = _overall_sentiment(frame)
    positive_share = counts.get("Positive", 0) / total * 100 if total else 0

    metric_columns = st.columns(4)
    metric_columns[0].metric("Comments analyzed", f"{total:,}")
    metric_columns[1].metric("Audience pulse", overall, f"{mean_score:+.2f} score")
    metric_columns[2].metric("Positive share", f"{positive_share:.1f}%")
    metric_columns[3].metric("Comment likes", _pretty_number(int(frame["likes"].sum())))

    left, right = st.columns([1, 1.6])
    with left:
        st.subheader("Sentiment mix")
        chart_data = pd.DataFrame(
            {"sentiment": list(COLORS), "comments": [int(counts.get(name, 0)) for name in COLORS]}
        )
        figure = px.pie(
            chart_data,
            values="comments",
            names="sentiment",
            hole=0.68,
            color="sentiment",
            color_discrete_map=COLORS,
        )
        figure.update_traces(textposition="outside", textinfo="percent+label", marker_line_width=0)
        figure.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), height=350)
        st.plotly_chart(figure, width="stretch")
    with right:
        st.subheader("Score distribution")
        figure = px.histogram(
            frame,
            x="compound",
            nbins=24,
            color="sentiment",
            color_discrete_map=COLORS,
            labels={"compound": "VADER compound score", "count": "Comments"},
        )
        figure.add_vline(x=-0.05, line_dash="dot", line_color="#8793A6")
        figure.add_vline(x=0.05, line_dash="dot", line_color="#8793A6")
        figure.update_layout(bargap=0.08, legend_title_text="", margin=dict(l=20, r=20, t=30, b=20), height=350)
        st.plotly_chart(figure, width="stretch")

    timeline = frame.dropna(subset=["published_at"]).copy()
    if not timeline.empty and timeline["published_at"].nunique() > 1:
        st.subheader("Audience mood over time")
        span = timeline["published_at"].max() - timeline["published_at"].min()
        period = "D" if span.days <= 90 else "ME"
        timeline = (
            timeline.set_index("published_at")
            .resample(period)["compound"]
            .agg(["mean", "count"])
            .dropna()
            .reset_index()
        )
        figure = px.line(
            timeline,
            x="published_at",
            y="mean",
            markers=True,
            hover_data={"count": True},
            labels={"published_at": "Published", "mean": "Average sentiment", "count": "Comments"},
        )
        figure.update_traces(line_color="#FF647C", line_width=3, marker_size=7)
        figure.add_hline(y=0, line_color="#566074", line_dash="dash")
        figure.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=320)
        st.plotly_chart(figure, width="stretch")

    keywords = top_keywords(frame["text"], limit=12)
    if not keywords.empty:
        st.subheader("What people mention")
        figure = px.bar(
            keywords.sort_values("mentions"),
            x="mentions",
            y="keyword",
            orientation="h",
            color="mentions",
            color_continuous_scale=["#343A50", "#FF647C"],
        )
        figure.update_layout(coloraxis_showscale=False, margin=dict(l=20, r=20, t=10, b=20), height=380)
        st.plotly_chart(figure, width="stretch")


def _comments_table(frame: pd.DataFrame) -> None:
    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        selected = st.multiselect("Sentiment", list(COLORS), default=list(COLORS))
    with search_col:
        search = st.text_input("Search comments", placeholder="Try a topic, phrase, or author…")

    filtered = frame[frame["sentiment"].isin(selected)].copy()
    if search:
        mask = filtered["text"].str.contains(search, case=False, na=False, regex=False)
        mask |= filtered["author"].str.contains(search, case=False, na=False, regex=False)
        filtered = filtered[mask]

    st.caption(f"Showing {len(filtered):,} of {len(frame):,} analyzed comments")
    display = filtered[["sentiment", "compound", "likes", "author", "text", "published_at", "is_reply"]].copy()
    display["compound"] = display["compound"].round(3)
    display["type"] = display.pop("is_reply").map({True: "Reply", False: "Top-level"})
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "sentiment": st.column_config.TextColumn("Sentiment"),
            "compound": st.column_config.NumberColumn("Score", format="%.3f"),
            "likes": st.column_config.NumberColumn("Likes", format="%d"),
            "text": st.column_config.TextColumn("Comment", width="large"),
            "published_at": st.column_config.DatetimeColumn("Published", format="YYYY-MM-DD HH:mm"),
        },
    )
    st.download_button(
        "Download analyzed CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="youtube_comment_sentiment.csv",
        mime="text/csv",
    )


def _show_sources(sources: list[RetrievedComment]) -> None:
    if not sources:
        return
    with st.expander(f"Evidence used · {len(sources)} comments"):
        for source in sources:
            st.markdown(f"**[{source.reference}] {source.author}** · {source.sentiment} · {source.likes} likes")
            st.write(source.text)


def _chat(frame: pd.DataFrame, gemini_key: str, model: str) -> None:
    st.subheader("Ask the audience")
    st.caption("Answers are grounded in a BM25-retrieved subset of the analyzed comments and include comment citations.")
    if not gemini_key:
        st.info("Add a Gemini API key in the sidebar to enable the LangChain chatbot.")

    if not st.session_state.messages:
        st.markdown("Try: *What do viewers like most?* · *What are the recurring complaints?* · *Summarize feature requests.*")
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _show_sources(st.session_state.chat_sources.get(index, []))

    question = st.chat_input("Ask about these comments…", disabled=not bool(gemini_key))
    if not question:
        return

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Reading the most relevant comments…"):
                answer, sources = answer_question(
                    frame,
                    question,
                    api_key=gemini_key,
                    model=model,
                    history=history,
                )
            st.markdown(answer)
            _show_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_sources[len(st.session_state.messages) - 1] = sources
        except Exception as exc:  # Gemini SDK errors vary by quota, key, and region.
            message = f"The chatbot could not answer: {exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})


def main() -> None:
    _inject_theme()
    _initialize_state()

    with st.sidebar:
        st.markdown("## Comment Pulse")
        st.caption("YouTube audience intelligence")
        st.divider()
        youtube_url = st.text_input("YouTube video", "https://www.youtube.com/watch?v=Xk2gwh0qG5s")
        youtube_key = "AIzaSyC0B8Mgvyu14q3LEOa0Kudgp6CYrZ-q9_8"
        max_comments = st.slider("Maximum comments", 100, 2_000, 500, step=100)
        include_replies = st.toggle("Include replies", value=True)
        order_label = st.selectbox("Comment order", ["Most relevant", "Newest first"])
        analyze = st.button("Analyze comments", type="primary", width="stretch")
        st.divider()
        st.markdown("### Chatbot")
        gemini_key = "AIzaSyAIop8woLPpZnWa9ysjyyodXM4oOAoQTD8"
        model = st.text_input("Gemini model", value="gemini-3.5-flash-lite")
        st.link_button("Get a Gemini API key ↗", "https://aistudio.google.com/app/apikey")
        st.caption("The default model supports Gemini's free tier, subject to Google's current quotas.")
        st.divider()
        st.caption("Sentiment uses VADER locally on CPU. It works best for English-language comments.")

    st.markdown('<div class="hero-kicker">Listen at scale</div>', unsafe_allow_html=True)
    st.title("Turn the comment section\ninto a signal.")
    st.markdown(
        '<p class="hero-copy">Measure audience sentiment, find the themes behind the numbers, '
        'and ask the comment section questions in plain language.</p>',
        unsafe_allow_html=True,
    )

    if analyze:
        try:
            if not youtube_key.strip():
                raise ValueError("Add a YouTube Data API key to start the analysis.")
            video_id = extract_video_id(youtube_url)
            progress = st.progress(0, text="Connecting to YouTube…")
            video = fetch_video_info(video_id, youtube_key.strip())
            raw_comments = fetch_comments(
                video_id,
                youtube_key.strip(),
                max_comments=max_comments,
                include_replies=include_replies,
                order="relevance" if order_label == "Most relevant" else "time",
                progress=lambda count: progress.progress(
                    min(count / max_comments, 0.95), text=f"Collected {count:,} comments…"
                ),
            )
            if not raw_comments:
                raise YouTubeAPIError("No public comments were returned for this video.")
            progress.progress(1.0, text="Scoring sentiment…")
            st.session_state.comments = analyze_comments(raw_comments)
            if not st.session_state.video or st.session_state.video.get("video_id") != video_id:
                st.session_state.messages = []
                st.session_state.chat_sources = {}
            st.session_state.video = video
            progress.empty()
            st.success(f"Analyzed {len(raw_comments):,} comments.")
        except (ValueError, YouTubeAPIError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    frame = st.session_state.comments
    video = st.session_state.video
    if frame is None or video is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="play">▶</div>
                <h3>Your audience dashboard starts here</h3>
                <p>Paste a public YouTube URL and add a YouTube Data API key in the sidebar, then choose Analyze comments.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    _video_header(video)
    st.write("")
    overview_tab, comments_tab, chat_tab = st.tabs(["Overview", "Comments", "Ask comments"])
    with overview_tab:
        _overview(frame)
    with comments_tab:
        _comments_table(frame)
    with chat_tab:
        _chat(frame, gemini_key.strip(), model.strip())


if __name__ == "__main__":
    main()
