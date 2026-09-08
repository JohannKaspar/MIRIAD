"""Rating app for the pre-registered human evaluation.

    streamlit run regeneration/rate_app.py

Mirrors quality_control/streamlit_app/: the same three criteria, worded as
there, one checkbox each per pair. Adds the forced choice on two-set screens.
Shows nothing that identifies an arm. Ratings are appended to
work/ratings.jsonl one screen at a time, so the session can stop and resume.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

W = Path("regeneration/work")
SCREENS = W / "screens_blinded.json"
RATINGS = W / "ratings.jsonl"

CRITERIA = [
    ("factual", "Factual", "The answer should be factually correct and accurate."),
    ("grounded", "Grounded in Passage", "The answer should be fully supported by the passage text."),
    ("relevant", "Relevant",
     "The Q&A should refer to medically relevant content. It is irrelevant if it contains specific "
     "details about a study's experimental design, statistical analysis methods, tables or figures, "
     "study dates, locations, funding sources, or other details that are not essential for "
     "understanding the key medical knowledge."),
]
FORCED_CHOICE = "Which set would you rather have in a retrieval corpus for medical question answering?"


def load_screens() -> list[dict]:
    return json.load(open(SCREENS))["screens"]


def rated_ids() -> set[str]:
    if not RATINGS.exists():
        return set()
    return {json.loads(l)["screen_id"] for l in RATINGS.read_text().splitlines() if l.strip()}


def main() -> None:
    st.set_page_config(page_title="MIRIAD pair rating", layout="wide")
    screens = load_screens()
    done = rated_ids()
    todo = [s for s in screens if s["screen_id"] not in done]

    st.sidebar.markdown(f"**Progress** {len(done)} / {len(screens)} screens")
    st.sidebar.progress(len(done) / max(len(screens), 1))
    with st.sidebar.expander("Criteria", expanded=False):
        for _, label, desc in CRITERIA:
            st.markdown(f"**{label}** — {desc}")

    if not todo:
        st.success("All screens rated. Thank you.")
        return

    screen = todo[0]
    sid = screen["screen_id"]
    n_sets = len(screen["sets"])
    st.caption(f"Screen {len(done) + 1} of {len(screens)}")

    st.markdown("#### Passage")
    st.markdown(
        f"<div style='max-height:340px;overflow-y:auto;padding:0.8rem 1rem;border:1px solid #ddd;"
        f"border-radius:6px;line-height:1.55'>{screen['passage']}</div>",
        unsafe_allow_html=True,
    )

    with st.form(key=f"form_{sid}", clear_on_submit=True):
        cols = st.columns(n_sets)
        checks: dict[str, bool] = {}
        for si, (col, s) in enumerate(zip(cols, screen["sets"]), 1):
            with col:
                st.markdown(f"### Set {si}")
                for pi, pair in enumerate(s["pairs"], 1):
                    st.markdown(f"**Q{pi}.** {pair['question']}")
                    st.markdown(f"{pair['answer']}")
                    ccols = st.columns([1, 1.7, 1])   # 'Grounded in Passage' must not truncate
                    for (ckey, label, _), ccol in zip(CRITERIA, ccols):
                        with ccol:
                            checks[f"set{si}_pair{pi}_{ckey}"] = st.checkbox(
                                label, key=f"{sid}_set{si}_pair{pi}_{ckey}")
                    st.markdown("---")

        choice = None
        if n_sets == 2:
            st.markdown(f"#### {FORCED_CHOICE}")
            choice = st.radio("Forced choice", ["Set 1", "Set 2", "No difference"],
                              index=None, horizontal=True, key=f"{sid}_choice",
                              label_visibility="collapsed")

        submitted = st.form_submit_button("Save and next", type="primary")
        if submitted:
            if n_sets == 2 and choice is None:
                st.error("Pick one of the three options before saving.")
                st.stop()
            record = {
                "screen_id": sid,
                "format": screen["format"],
                "criteria": checks,
                "forced_choice": {"Set 1": 1, "Set 2": 2, "No difference": 0}.get(choice) if choice else None,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            with RATINGS.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            st.rerun()


if __name__ == "__main__":
    main()
