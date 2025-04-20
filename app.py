import streamlit as st
from utils import load_model, preprocess_text
import nltk

nltk.download('stopwords')
model = load_model('./models/best_model.joblib')

min_words_number = 100

def check_generated_text(text):
    filtered_text = preprocess_text(text)
    prediction = model.predict([filtered_text])
    return not int(prediction[0])

# Load styles
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.title("Generated Text Checker")

# Initialize session state
if "check_clicked" not in st.session_state:
    st.session_state.check_clicked = False

# Use a form to isolate the check action
with st.form("text_check_form"):
    user_input = st.text_area(
        f"Enter text to check",
        height=400,
        placeholder=f"Paste your generated text here... it should be at least {min_words_number} words"
    )
    submitted = st.form_submit_button("Check text")

# Handle form submission
if submitted:
    st.session_state.check_clicked = True

# Only run check when button is clicked
if st.session_state.check_clicked:
    with st.spinner("Checking text..."):
        current_length = len(user_input.split())
        
        if current_length >= min_words_number:
            result = check_generated_text(user_input)
            if result:
                st.info("✅ The text appears to be human-written!")
            else:
                st.info("🤖 The text appears to be AI-generated.")
        else:
            st.warning(f"Please enter at least {min_words_number} words.")
                
    # Reset check state
    st.session_state.check_clicked = False