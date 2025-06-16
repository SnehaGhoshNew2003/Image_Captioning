import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import preprocess_input
import pickle

# Define a dummy NotEqual class for loading the model if it was saved with it.
# This is a workaround if the custom_objects in model.load_model requires it.
class NotEqual(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)
    def call(self, inputs):
        # This layer might be used for specific custom loss or metrics,
        # but for inference, it might not be directly used in the forward pass.
        # If your model truly depends on a NotEqual operation, you would implement it here.
        # For a placeholder, we'll assume it's not critical for the predict function.
        return inputs # Placeholder, might need actual implementation

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model(
            "best_model.h5",
            custom_objects={'NotEqual': NotEqual} # Use the defined NotEqual class
        )
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.stop() # Stop the app if essential files can't be loaded

# Load Xception for feature extraction
@st.cache_resource
def get_feature_extractor():
    base_model = tf.keras.applications.Xception(include_top=False, pooling="avg")
    return base_model

# Generate caption
def generate_caption(model, tokenizer, photo, max_length=35):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # The corrected line: verbose is passed as a keyword argument
        y_pred = model.predict([photo, sequence], verbose=0)

        y_pred = np.argmax(y_pred)
        word = tokenizer.index_word.get(y_pred)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Streamlit UI
st.title("🖼️ Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("⏳ Generating caption...")

    # Preprocess image
    img = image.resize((299, 299))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Extract image features
    feature_extractor = get_feature_extractor()
    photo_features = feature_extractor.predict(img_array, verbose=0)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Generate caption
    caption = generate_caption(model, tokenizer, photo_features)
    #st.success(f"📝 Generated Caption: **{caption}**")
    st.markdown(f"""
<div style="background-color:#d4edda; color:#155724; padding:15px; border-radius:8px; border:1px solid #c3e6cb; margin-top:20px">
    <span style="font-size:28px; font-weight:600;">📝 Generated Caption: {caption}</span>
</div>
""", unsafe_allow_html=True)
