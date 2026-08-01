import streamlit as st
import requests

# The URL where our FastAPI backend is running
API_URL = "https://spam-email-classifier-x74c.onrender.com/predict"  # Update this to your deployed API URL if different

st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("📧 AI Spam Email Classifier")
st.write("Enter a message below to check if it's spam or legitimate (ham).")

# A multi-line text input box for the user to type/paste their message
user_input = st.text_area("Message text", height=150, placeholder="Type or paste a message here...")

if st.button("Check Message"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:
        try:
            response = requests.post(API_URL, json={"text": user_input})

            if response.status_code == 200:
                result = response.json()

                if result["prediction"] == "spam":
                    st.error(f"🚫 This looks like SPAM (confidence: {result['confidence']*100:.1f}%)")
                else:
                    st.success(f"✅ This looks like HAM/legitimate (confidence: {result['confidence']*100:.1f}%)")

                with st.expander("See cleaned text (what the model actually analyzed)"):
                    st.code(result["cleaned_text"])
            else:
                st.error(f"API returned an error (status code: {response.status_code})")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running.")