"""
app.py
------
Streamlit Web Application for Human Activity Recognition.
Features:
- Home: Displays project overview, supported classes, and model accuracy.
- Live Webcam: Real-time pose prediction stream.
- Upload Video: File upload with manual analysis button and prediction charts.
"""

from datetime import datetime
import json
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

MODEL_DIR = Path("models")
DATA_DIR = Path("processed_data")

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    from predict import ActivityPredictor

    return ActivityPredictor(str(MODEL_DIR), str(DATA_DIR))


def page_home():
    st.title("🏃 Human Activity Recognition System")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Model Performance")
        st.metric(label="Test Accuracy", value="56.86%")
        
        # Load classes dynamically if metadata exists
        classes = ["sit", "run", "wave", "jumping", "walk"]
        metadata_path = DATA_DIR / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                    classes = meta.get("classes", classes)
            except Exception:
                pass

        st.metric(label="Total Activity Classes", value=len(classes))

    with col2:
        st.subheader("🎯 Recognized Activities")
        for cls_name in classes:
            st.markdown(f"- **{cls_name.capitalize()}**")

    st.markdown("---")
    st.subheader("📌 Project Navigation")
    st.write(
        "Use the sidebar on the left to navigate:\n"
        "* **Live Webcam:** Real-time activity detection directly through your webcam.\n"
        "* **Upload Video:** Test uploaded video files (.mp4, .avi, .mov) with confidence scores and history tracking."
    )


def page_live_webcam():
    st.header("Live Webcam Recognition")
    st.write("Turn on your camera to predict activity in real time.")

    run_camera = st.checkbox("Start Webcam", value=False)

    frame_placeholder = st.empty()
    banner_placeholder = st.empty()
    chart_placeholder = st.empty()

    st.subheader("Prediction History")
    history_placeholder = st.empty()

    if "webcam_history" not in st.session_state:
        st.session_state["webcam_history"] = []

    if run_camera:
        try:
            predictor = load_predictor()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Ensure your camera is connected.")
            return

        last_logged_time = 0

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame from camera.")
                break

            _ = predictor.push_frame(frame)
            res = predictor.predict_live()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            if res is not None:
                label, confidence, probs = res

                banner_placeholder.success(
                    f"**Predicted activity:** {label} ({confidence * 100:.1f}% confidence)"
                )

                df_probs = pd.DataFrame(
                    {"Activity": predictor.class_names, "Probability": probs}
                ).set_index("Activity")
                chart_placeholder.bar_chart(df_probs)

                current_time = datetime.now()
                if (current_time.timestamp() - last_logged_time) >= 2.0:
                    st.session_state["webcam_history"].insert(
                        0,
                        {
                            "timestamp": current_time.strftime("%H:%M:%S"),
                            "prediction": label,
                            "confidence": f"{confidence * 100:.1f}%",
                        },
                    )
                    last_logged_time = current_time.timestamp()

            if st.session_state["webcam_history"]:
                history_df = pd.DataFrame(st.session_state["webcam_history"])
                history_placeholder.dataframe(history_df)

        cap.release()
        frame_placeholder.empty()


def page_upload_video():
    st.header("Upload Video")

    if "video_history" not in st.session_state:
        st.session_state["video_history"] = []

    uploaded_file = st.file_uploader(
        "Upload a video file", type=["mp4", "avi", "mov", "mpeg4"]
    )

    if uploaded_file is not None:
        temp_dir = Path("outputs/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / uploaded_file.name

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(str(video_path))

        if st.button("Analyze Video"):
            with st.spinner("Analyzing video..."):
                try:
                    predictor = load_predictor()
                    label, confidence, probs = predictor.predict_video(str(video_path))

                    st.success(
                        f"**Predicted activity:** {label} ({confidence * 100:.1f}% confidence)"
                    )

                    df_probs = pd.DataFrame(
                        {"Activity": predictor.class_names, "Probability": probs}
                    ).set_index("Activity")
                    st.bar_chart(df_probs)

                    st.session_state["video_history"].insert(
                        0,
                        {
                            "file": uploaded_file.name,
                            "prediction": label,
                            "confidence": f"{confidence * 100:.1f}%",
                        },
                    )
                except Exception as e:
                    st.error(f"Error processing video: {e}")

    if st.session_state["video_history"]:
        st.subheader("Prediction History")
        st.dataframe(pd.DataFrame(st.session_state["video_history"]))


def main():
    st.sidebar.title("Human Activity Recognition")
    choice = st.sidebar.radio("Select Page", ["Home", "Live Webcam", "Upload Video"])

    if choice == "Home":
        page_home()
    elif choice == "Live Webcam":
        page_live_webcam()
    elif choice == "Upload Video":
        page_upload_video()


if __name__ == "__main__":
    main()