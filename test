import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import logging
import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
import plotly.graph_objects as go
from PIL import Image
import cv2
from deepface import DeepFace
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp
import time

# Initialize logging
try:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='app_debug.log'
    )
except Exception as e:
    logging.basicConfig(level=logging.DEBUG)  # Fallback to console logging
    logging.error(f"Failed to initialize file logging: {str(e)}")
logger = logging.getLogger(__name__)

@st.cache_data
def load_data() -> pd.DataFrame:
    data = {
        "Hemisphere": ["Left Brain (Purple)"] * 14 + ["Right Brain (Yellow)"] * 11,
        "Element": [
            "Voiceover", "Split-screen effect", "Audio repetitions", "Words obtruding",
            "Abstracted body part", "Freeze-frame effect", "Flatness", 
            "Highly rhythmic soundtrack", "Abstracted product", "Product-centricity",
            "Self-consciousness", "Facial frontality", "Empty smile", "Monologue",
            "Music with discernible melody", "Something out of the ordinary",
            "One scene unfolding with progression", "A clear sense of place",
            "Characters with vitality/agency", "Implicit and unspoken communication",
            "Dialogue", "Spontaneous change in facial expression", "Distinctive accents",
            "People touching", "Animal(s)"
        ],
        "X_Value": [-0.05, 0.0, -0.1, -0.05, -0.05, -0.15, -0.15, -0.1, -0.1, -0.2,
                    -0.2, -0.25, -0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.3, 0.22, 0.15,
                    0.25, 0.2, 0.35, 0.5],
        "Y_Value": [0.1, 0.05, 0.05, 0.0, 0.0, 0.0, -0.05, -0.05, -0.1, -0.15,
                    -0.1, -0.15, -0.2, 0.0, 0.2, 0.18, 0.15, 0.12, 0.22, 0.15,
                    0.1, 0.1, 0.05, 0.08, 0.25]
    }
    return pd.DataFrame(data)

def process_image(uploaded_file) -> tuple[Image.Image, np.ndarray]:
    try:
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        # Handle different image channels (grayscale, RGBA, etc.)
        if len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB
            pass
        else:
            raise ValueError(f"Unexpected image format: {image_array.shape}")

        return image, image_array
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

def analyze_image(detected_attributes: list[str], df: pd.DataFrame) -> tuple[float, float]:
    if not detected_attributes:
        logger.warning("No attributes detected")
        return 0.0, 0.0
    
    left_brain_df = df[df['Hemisphere'] == 'Left Brain (Purple)']
    right_brain_df = df[df['Hemisphere'] == 'Right Brain (Yellow)']
    
    left_score = sum(abs(left_brain_df[left_brain_df['Element'] == attr]['Y_Value'].iloc[0])
                    for attr in detected_attributes if attr in left_brain_df['Element'].values)
    
    right_score = sum(right_brain_df[right_brain_df['Element'] == attr]['Y_Value'].iloc[0]
                     for attr in detected_attributes if attr in right_brain_df['Element'].values)
            
    return left_score, right_score

def create_pie_chart(left_count: int, right_count: int) -> go.Figure:
    labels = ['Left Brain', 'Right Brain']
    values = [left_count, right_count]
    colors = ['rgba(128, 0, 128, 0.7)', 'rgba(255, 223, 0, 0.7)']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=14),
        hovertemplate="<b>%{label}</b><br>" +
                      "Count: %{value}<br>" +
                      "Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title={
            'text': "Dominance Distribution",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        width=600,
        height=400
    )
    
    return fig

class EmotionDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.emotion_sequences = []
        self.time_stamps = []
        self.max_sequence_length = 100

    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

    def analyze_facial_asymmetry(self, image):
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return 0.0
        
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = np.mean([(landmarks[33].x, landmarks[33].y), (landmarks[133].x, landmarks[133].y)], axis=0)
        right_eye = np.mean([(landmarks[362].x, landmarks[362].y), (landmarks[263].x, landmarks[263].y)], axis=0)
        
        asymmetry_score = np.abs(left_eye[0] - right_eye[0]) + np.abs(left_eye[1] - right_eye[1])
        return asymmetry_score

    def detect_micro_expressions(self, emotion_sequence, time_stamps):
        if len(emotion_sequence) < 2:
            return False
            
        rapid_changes = 0
        for i in range(1, len(emotion_sequence)):
            time_diff = time_stamps[i] - time_stamps[i-1]
            if time_diff < 0.5 and emotion_sequence[i] != emotion_sequence[i-1]:
                rapid_changes += 1
                
        return rapid_changes > 2

    def analyze_emotion_coherence(self, face_analysis):
        try:
            emotion = face_analysis['dominant_emotion']
            emotion_scores = face_analysis['emotion']
            if emotion == 'happy' and emotion_scores.get('sad', 0) > 0.3:
                return False
            if emotion == 'sad' and emotion_scores.get('happy', 0) > 0.3:
                return False
            return True
        except Exception as e:
            logger.error(f"Emotion coherence analysis error: {str(e)}")
            return True

    def map_emotions_to_attributes(self, face_analysis, authenticity_score):
        emotion_attributes = set()
        try:
            emotion = face_analysis.get('dominant_emotion', 'unknown')
            emotion_scores = face_analysis.get('emotion', {})
            
            if authenticity_score < 0.7:
                emotion_attributes.add("Empty smile")
                emotion_attributes.add("Self-consciousness")
            if authenticity_score < 0.4:
                emotion_attributes.add("Product-centricity")

            if emotion == 'neutral':
                emotion_attributes.add("Words obtruding")
            
            if authenticity_score > 0.7:
                if emotion in ['happy', 'surprise']:
                    emotion_attributes.add("Characters with vitality/agency")
                    if emotion_scores.get('happy', 0) > 0.8:
                        emotion_attributes.add("Dialogue")

                if emotion in ['sad', 'fear', 'angry']:
                    emotion_attributes.add("Implicit and unspoken communication")
                    if emotion_scores.get(emotion, 0) > 0.7:
                        emotion_attributes.add("Something out of the ordinary")

                if emotion in ['happy', 'surprise', 'fear', 'angry']:
                    emotion_attributes.add("Spontaneous change in facial expression")

            return list(emotion_attributes)
        except Exception as e:
            logger.error(f"Emotion-to-attributes mapping error: {str(e)}")
            return []

class AttributeDetector:
    def __init__(self):
        try:
            self.base_model = ResNet50(weights='imagenet', include_top=False)
            logger.info("ResNet50 successfully initialized")
        except Exception as e:
            logger.error(f"ResNet50 initialization error: {str(e)}")
            st.error("Error loading base model")
            self.base_model = None
        
        self.emotion_detector = EmotionDetector()
        self.batch_size = 32

    def analyze_emotions(self, image: np.ndarray) -> tuple[list[str], dict]:
        try:
            face_analysis = DeepFace.analyze(
                img_array=image,  # Ensure it's numpy array
                actions=['emotion'],
                enforce_detection=False
            )
            
            if isinstance(face_analysis, list):
                face_analysis = face_analysis[0]
            
            current_time = time.time()
            self.emotion_detector.emotion_sequences.append(face_analysis['dominant_emotion'])
            self.emotion_detector.time_stamps.append(current_time)
            
            if len(self.emotion_detector.emotion_sequences) > self.emotion_detector.max_sequence_length:
                self.emotion_detector.emotion_sequences = self.emotion_detector.emotion_sequences[-self.emotion_detector.max_sequence_length:]
                self.emotion_detector.time_stamps = self.emotion_detector.time_stamps[-self.emotion_detector.max_sequence_length:]
            
            asymmetry_score = self.emotion_detector.analyze_facial_asymmetry(image)
            has_micro_expressions = self.emotion_detector.detect_micro_expressions(
                self.emotion_detector.emotion_sequences,
                self.emotion_detector.time_stamps
            )
            is_coherent = self.emotion_detector.analyze_emotion_coherence(face_analysis)
            
            authenticity_score = (
                (0.4 * (asymmetry_score > 0.1)) +
                (0.3 * (not has_micro_expressions)) +
                (0.3 * is_coherent)
            )
                
            emotion_attributes = self.emotion_detector.map_emotions_to_attributes(
                face_analysis, authenticity_score
            )
            
            return emotion_attributes, {
                'dominant_emotion': face_analysis['dominant_emotion'],
                'authenticity_score': authenticity_score,
                'has_micro_expressions': has_micro_expressions,
                'is_coherent': is_coherent,
                'asymmetry_score': asymmetry_score
            }
        except Exception as e:
            logger.error(f"Emotion analysis error: {str(e)}")
            return [], {
                'dominant_emotion': 'unknown',
                'authenticity_score': 0.0,
                'has_micro_expressions': False,
                'is_coherent': False,
                'asymmetry_score': 0.0
            }

    def detect_all_attributes(self, image: np.ndarray) -> tuple[list[str], dict]:
        try:
            attributes = set()
            emotion_attrs, emotion_details = self.analyze_emotions(image)
            attributes.update(emotion_attrs)
            
            if self.base_model is not None:
                img_resized = cv2.resize(image, (224, 224))
                img_array = np.expand_dims(img_resized, axis=0)
                img_array = img_to_array(img_array)
                
                features = self.base_model.predict(
                    img_array,
                    batch_size=self.batch_size,
                    verbose=0
                )
                
                feature_mean = np.mean(features)
                if feature_mean > 0.5:
                    attributes.add("A clear sense of place")
                if feature_mean > 0.7:
                    attributes.add("One scene unfolding with progression")
                if np.max(features) > 0.9:
                    attributes.add("Something out of the ordinary")
                    
            return list(attributes), emotion_details
        except Exception as e:
            logger.error(f"Attribute detection error: {str(e)}")
            return [], {
                'dominant_emotion': 'unknown',
                'authenticity_score': 0.0,
                'has_micro_expressions': False,
                'is_coherent': False,
                'asymmetry_score': 0.0
            }

def main():
    st.set_page_config(
        page_title="WoodVision",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("WoodVision 🧠")
    st.markdown("""
        *This analyzer is based on the science of Orlando Wood and categorizes images based on the left and right hemisphere attributes with a special focus on emotions.*
    """)
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    try:
        detector = AttributeDetector()
        df = load_data()
        
        with st.sidebar:
            st.header("Settings")
            show_debug = st.checkbox("Show debug information", False)
            clear_cache = st.button("Clear cache")
            
            if clear_cache:
                st.session_state.processed_files.clear()
                st.session_state.results.clear()
                st.cache_data.clear()
                st.success("Cache cleared")
    
        uploaded_files = st.file_uploader(
            "Choose images (PNG, JPG, JPEG)", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True
        )
    
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    with st.expander(f"Analyzing: {uploaded_file.name}", expanded=True):
                        try:
                            image, image_array = process_image(uploaded_file)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(image_array, caption=uploaded_file.name, use_column_width=True)
                            
                            with col2:
                                with st.spinner("Performing analysis..."):
                                    detected_attributes, emotion_details = detector.detect_all_attributes(image_array)
                                    
                                    st.subheader("Emotion Analysis")
                                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                    with metrics_col1:
                                        st.metric("Dominant Emotion", emotion_details['dominant_emotion'].capitalize())
                                    with metrics_col2:
                                        st.metric("Authenticity", f"{emotion_details['authenticity_score']:.2f}")
                                    with metrics_col3:
                                        st.metric("Emotion Coherence", "✓" if emotion_details['is_coherent'] else "✗")
                                    
                                    if emotion_details['has_micro_expressions']:
                                        st.warning("⚠️ Microexpressions detected")
                                    
                                    if show_debug:
                                        st.json(emotion_details)
                            
                            left_score, right_score = analyze_image(detected_attributes, df)
                            dominant = "Left" if left_score > right_score else "Right"
                            
                            st.subheader("Brain Hemisphere Analysis")
                            score_col1, score_col2, score_col3 = st.columns(3)
                            with score_col1:
                                st.metric("Left Brain", f"{left_score:.2f}")
                            with score_col2:
                                st.metric("Right Brain", f"{right_score:.2f}")
                            with score_col3:
                                st.metric("Dominant Side", dominant)

                            st.subheader("Detected Attributes")
                            attributes_data = []
                            for attr in detected_attributes:
                                attr_row = df[df['Element'] == attr].iloc[0]
                                hemisphere = attr_row['Hemisphere']
                                y_value = abs(attr_row['Y_Value'])
                                
                                attributes_data.append({
                                    'Attribute': attr,
                                    'Hemisphere': 'Left' if 'Left' in hemisphere else 'Right',
                                    'Weight': f"{y_value:.3f}"
                                })
                            
                            if attributes_data:
                                attrs_df = pd.DataFrame(attributes_data)
                                attrs_df = attrs_df.sort_values('Weight', ascending=False)
                                
                                left_attrs = attrs_df[attrs_df['Hemisphere'] == 'Left']
                                right_attrs = attrs_df[attrs_df['Hemisphere'] == 'Right']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("Left Brain Attributes:")
                                    if not left_attrs.empty:
                                        st.dataframe(left_attrs[['Attribute', 'Weight']], hide_index=True)
                                    else:
                                        st.write("No attributes detected")
                                
                                with col2:
                                    st.write("Right Brain Attributes:")
                                    if not right_attrs.empty:
                                        st.dataframe(right_attrs[['Attribute', 'Weight']], hide_index=True)
                                    else:
                                        st.write("No attributes detected")
                                
                                st.write("Overall Evaluation:")
                                st.write(f"- Left Brain Total: {left_score:.3f}")
                                st.write(f"- Right Brain Total: {right_score:.3f}")
                                st.write(f"- Dominant Hemisphere: {dominant} (Difference: {abs(left_score - right_score):.3f})")
                            else:
                                st.warning("No attributes detected")
                            
                            st.session_state.results.append({
                                'filename': uploaded_file.name,
                                'dominant': dominant,
                                'left_score': left_score,
                                'right_score': right_score,
                                'detected_attributes': ", ".join(detected_attributes),
                                'emotion': emotion_details['dominant_emotion'],
                                'authenticity': emotion_details['authenticity_score']
                            })
                            
                            st.session_state.processed_files.add(uploaded_file.name)
                            
                        except Exception as e:
                            logger.error(f"Error analyzing {uploaded_file.name}: {str(e)}")
                            st.error(f"Error analyzing {uploaded_file.name}: {str(e)}")
                            continue
            
            if st.session_state.results:
                st.header("Overall Evaluation")
                results_df = pd.DataFrame(st.session_state.results)
                
                left_dominant = len(results_df[results_df['dominant'] == 'Left'])
                right_dominant = len(results_df[results_df['dominant'] == 'Right'])
                
                try:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = create_pie_chart(left_dominant, right_dominant)
                        st.plotly_chart(fig)
                    
                    with col2:
                        if 'emotion' in results_df.columns:
                            emotion_counts = results_df['emotion'].value_counts()
                            st.bar_chart(emotion_counts)
                    
                    total_images = len(results_df)
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Average Authenticity", f"{results_df['authenticity'].mean():.2f}")
                    with stats_col2:
                        st.metric("Left Brain", f"{(left_dominant/total_images)*100:.1f}%")
                    with stats_col3:
                        st.metric("Right Brain", f"{(right_dominant/total_images)*100:.1f}%")
                    
                    if show_debug:
                        st.subheader("Detailed Results")
                        st.dataframe(results_df)
                    
                except Exception as e:
                    logger.error(f"Error generating overall evaluation: {str(e)}")
                    st.error("Error generating overall evaluation")
    
    except Exception as e:
        logger.error(f"Critical application error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
