"""Run with `python -m streamlit run demo.py` from the project directory."""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from eeg_data import CLASS_LABELS, FEATURE_NAMES, HISTORY, N_CLASSES, N_FEATURES
from eeg_model import load_checkpoint, predict_window


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


@st.cache_resource
def get_model(path, modified_ns):
    return load_checkpoint(path)


@st.cache_data
def load_demo_windows(path, modified_ns):
    with np.load(path, allow_pickle=False) as data:
        if tuple(data["feature_names"]) != FEATURE_NAMES:
            raise ValueError("The demo window file has the wrong feature order.")
        windows = data["windows"]
        records = data["record_id"]
        ends = data["epoch_end_sec"]
        archive_hash = data["archive_sha256"].item()
    if windows.shape != (len(records), HISTORY, N_FEATURES) or records.shape != ends.shape:
        raise ValueError("The demo window file has invalid shapes.")
    return windows, records, ends, archive_hash


def read_upload(file):
    if Path(file.name).suffix.lower() == ".npy":
        return np.load(file, allow_pickle=False)
    return np.loadtxt(file, delimiter=",", dtype=np.float32, ndmin=2)


def state_plot(probabilities, threshold):
    end_times = np.arange(-4.5, 0.5, 0.5)
    fig, ax = plt.subplots(figsize=(8, 2.8), layout="constrained")
    colors = np.where(probabilities >= threshold, "#284e9b", "#c88229")
    ax.bar(end_times, probabilities, width=0.37, color=colors)
    ax.axhline(threshold, color="#606671", linestyle="--", label="Sleep threshold")
    ax.set(xticks=end_times, xlabel="Minutes from prediction time · epoch end",
           ylabel="P(Asleep)", ylim=(0, 1.06))
    ax.legend(frameon=False, loc="upper left")
    return fig


def forecast_plot(probabilities):
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    bins = np.arange(N_CLASSES)
    colors = ["#284e9b"] * (N_CLASSES - 1) + ["#7f8a96"]
    ax.barh(bins, probabilities * 100, color=colors)
    ax.set(yticks=bins, yticklabels=CLASS_LABELS, xlabel="Probability (%)",
           xlim=(0, max(probabilities) * 112))
    ax.invert_yaxis()
    return fig


def time_label(seconds):
    hours, tail = divmod(int(seconds), 3600)
    minutes, seconds = divmod(tail, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def api_json(base_url, path, payload=None):
    if payload is None:
        request = Request(base_url + path)
    else:
        request = Request(base_url + path, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def main():
    st.set_page_config(page_title="Sleep–Wake Forecast", layout="centered")
    st.title("Sleep–Wake Forecast")
    api_url = os.getenv("SLEEPOPS_API_URL", "").rstrip("/")
    checkpoint = ARTIFACTS / "best_checkpoint.pt"
    try:
        if api_url:
            details = api_json(api_url, "/model")
        else:
            model, mean, scale, details = get_model(str(checkpoint), checkpoint.stat().st_mtime_ns)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        st.error(f"Could not connect to the model. {error}")
        st.stop()

    source = st.radio("EEG source", ["Held-out recording", "Upload a feature window"], horizontal=True)
    if source == "Held-out recording":
        example_path = ARTIFACTS / "demo_windows.npz"
        try:
            windows, records, ends, archive_hash = load_demo_windows(
                str(example_path), example_path.stat().st_mtime_ns,
            )
            if archive_hash != details["archive_sha256"]:
                raise ValueError("These demo windows belong to a different EEG archive.")
            if not api_url and not set(records).issubset(details["test_records"]):
                raise ValueError("These demo windows do not match the checkpoint’s held-out split.")
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Could not load the held-out windows. {error}")
            st.stop()
        left, right = st.columns(2)
        record = left.selectbox("Test recording", sorted(set(records)))
        rows = np.flatnonzero(records == record)
        rows = rows[np.argsort(ends[rows])]
        selected = right.selectbox("Window end from recording start", rows.tolist(),
                                   index=len(rows) // 2, format_func=lambda row: time_label(ends[row]))
        window = windows[selected]
    else:
        upload = st.file_uploader("Prepared EEG features (.npy or headerless .csv)", type=["npy", "csv"])
        st.caption("Use 10 chronological 30-second rows. The last row ends at the prediction time.")
        with st.expander("Feature column order"):
            st.code(", ".join(FEATURE_NAMES), language="text")
        if upload is None:
            st.stop()
        try:
            window = read_upload(upload)
        except (OSError, ValueError, TypeError, EOFError) as error:
            st.error(f"Could not read this feature window. {error}")
            st.stop()

    try:
        if api_url:
            result = api_json(api_url, "/v1/predict", {"features": window.tolist()})
            p_state = np.asarray(result["asleep_probabilities"])
            forecast = result["forecast"]
            p_forecast = np.asarray([row["probability"] for row in forecast["bins"]]) if forecast else None
            threshold = result["state_threshold"]
            st.caption(f"Model release {result['release_id'][:12]}")
        else:
            p_state, p_forecast = predict_window(model, mean, scale, window)
            threshold = details["state_threshold"]
    except (OSError, ValueError, TypeError, KeyError) as error:
        st.error(str(error))
        st.stop()
    asleep = p_state[-1] >= threshold
    st.subheader("Asleep now" if asleep else "Awake now")
    st.caption(f"Current probability of sleep {p_state[-1]:.1%}")
    st.subheader("Sleep–wake timeline")
    fig = state_plot(p_state, threshold)
    st.pyplot(fig)
    plt.close(fig)

    if asleep:
        most_likely = int(p_forecast.argmax())
        if most_likely == N_CLASSES - 1:
            st.write("Most likely outcome is no sustained awakening before 60 minutes.")
        else:
            low = most_likely * 5
            st.write(f"Most likely sustained awakening in {low}–{low + 5} minutes.")
        st.subheader("Awakening forecast")
        st.caption(f"Probability of a sustained awakening within the hour {1 - p_forecast[-1]:.1%}")
        fig = forecast_plot(p_forecast)
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("All 13 probabilities"):
            rows = ["| Time to sustained awakening | Probability |", "| --- | ---: |"]
            rows.extend(f"| {label} | {p:.2%} |" for label, p in zip(CLASS_LABELS, p_forecast))
            st.markdown("\n".join(rows))


if __name__ == "__main__":
    main()
