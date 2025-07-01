import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.layers import TFSMLayer
import pandas as pd

@st.cache_resource
def load_models():
    classifier = keras.models.load_model("finalModel.h5")
    vectorizer = TFSMLayer("vecmodel", call_endpoint="serving_default")
    return classifier, vectorizer

classifier, vectorizer = load_models()

# Initialize history in session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Custom CSS for big font and colored labels
st.markdown("""
<style>
.big-font {
    font-size: 36px !important;
    font-weight: bold !important;
    margin-bottom: 0.25em;
}
.label-roast {
    color: #e63946;  /* red */
    font-weight: bold;
}
.label-toast {
    color: #2a9d8f;  /* teal */
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🧠 Roast vs. Toast Classifier</p>', unsafe_allow_html=True)
st.write("Enter a message below, and the model will classify it as a roast or a toast!")

user_input = st.text_area("💬 Enter your text here:", height=150)

if st.button("🔥 Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classifying... 🔍"):
            try:
                input_tensor = tf.constant([[user_input]], dtype=tf.string)  # shape (1,1)
                vectorized_input = vectorizer(inputs=input_tensor)

                if isinstance(vectorized_input, dict):
                    vectorized_input = list(vectorized_input.values())[0]

                prediction = classifier.predict(vectorized_input)[0][0]

                label = "Roast" if prediction > 0.5 else "Toast"
                confidence = prediction if label == "Roast" else 1 - prediction

                # Show colored label with emoji
                if label == "Roast":
                    st.markdown(f'<h2 class="label-roast">🔥 Roast</h2>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<h2 class="label-toast">🥳 Toast</h2>', unsafe_allow_html=True)

                st.write(f"Confidence: `{confidence:.2%}`")

                # Append to history
                st.session_state.history.append({
                    'prompt': user_input,
                    'label': label,
                    'confidence': f"{confidence:.2%}"
                })

            except Exception as e:
                st.error(f"💥 Error while classifying: {e}")

if st.session_state.history:
    st.markdown("### 📜 Prediction History")

    # Show history dataframe
    df = pd.DataFrame(st.session_state.history)

    # Add emoji column for fun
    df['emoji'] = df['label'].apply(lambda x: "🔥" if x == "Roast" else "🥳")

    # Reorder columns for display
    df = df[['emoji', 'prompt', 'label', 'confidence']]

    st.dataframe(df.rename(columns={
        'emoji': '',
        'prompt': 'Prompt',
        'label': 'Prediction',
        'confidence': 'Confidence'
    }))
