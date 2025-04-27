import streamlit as st
import torch
from src.model import get_model
from src.utils import compute_distance

st.title("Semantic Sentence Similarity (Pro Version)")

model = get_model()

sent1 = st.text_input("Enter first sentence:")
sent2 = st.text_input("Enter second sentence:")

if sent1 and sent2:
    embeddings = model.encode([sent1, sent2], convert_to_tensor=True)
    distance = compute_distance(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))

    st.write(f"Distance between sentences: {distance:.4f}")
    if distance < 1.0:
        st.success("✅ Sentences are similar!")
    else:
        st.error("❌ Sentences are different.")
import os
os.environ["PORT"] = os.getenv("PORT", "8501")

import streamlit as st
# your other imports

