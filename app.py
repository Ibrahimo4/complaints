import numpy as np
import spacy
import logging
import time
from gensim.models import Word2Vec
import streamlit as st
from tensorflow.keras.models import load_model
import sqlite3  # Database
import streamlit_authenticator as stauth

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to SQLite database
conn = sqlite3.connect('user_details.db')
cursor = conn.cursor()

# Create a table to store user details
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT,
        issue TEXT,
        company_public_response TEXT,
        company_name TEXT,
        tags TEXT,
        submission_location TEXT,
        company_response TEXT,
        timely_response TEXT
    )
''')
conn.commit()

# Sidebar styling with custom CSS
st.markdown("""
    <style>
    .css-18e3th9 {
        background-color: #DCDCDC !important; /* Light gray background */
        color: #4B0082; /* Indigo text color */
    }
    .css-1d391kg {
        font-family: 'Arial', sans-serif;
        color: #4B0082; /* Indigo text for sidebar elements */
    }
    </style>
    """, unsafe_allow_html=True)

# Function to insert user details into the database
def insert_user_details(product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response):
    cursor.execute('''
        INSERT INTO user_details (
            product_name,
            issue,
            company_public_response,
            company_name,
            tags,
            submission_location,
            company_response,
            timely_response
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response))
    conn.commit()

# Function to get user details from the database
def get_user_details():
    cursor.execute('''
        SELECT * FROM user_details
    ''')
    user_details = cursor.fetchall()
    return user_details

# Language selection
language = st.sidebar.radio("Select Language / اختر اللغة", ("English", "العربية"))

# Title
title = "Consumer Complaint Disputer 🗳️" if language == "English" else "نظام شكاوى المستهلك 🗳️"
st.markdown(f"<h1 style='text-align: center; color: #4B0082; font-family: cursive;'>{title}</h1>", unsafe_allow_html=True)

# Warning message
if language == "English":
    st.warning("We appreciate you utilizing our AI-driven customer dispute system. Ensure the provided information is genuine.")
else:
    st.warning("نقدر استخدامك لنظام النزاعات المعتمد على الذكاء الاصطناعي. يرجى التأكد من أن المعلومات المقدمة حقيقية.")

# Define login credentials (example structure)
users = {
    "usernames": {
        "user1": {
            "name": "John Doe",
            "password": "password123"
        }
    }
}
authenticator = stauth.Authenticate(users, "ComplaintApp", "abcdef", cookie_expiry_days=1)

# Sidebar for Login/Signup
st.sidebar.markdown("## Login / Signup")
name, authentication_status, username = authenticator.login(location="sidebar", key="login")

# Dashboard and Complaint Section if authenticated
if authentication_status:
    authenticator.logout("Logout", "sidebar", key="logout")
    st.sidebar.markdown(f"Welcome, **{name}**!" if language == "English" else f"مرحباً، **{name}**!")

    # About Dataset button
    about_text = "About the Dataset we have trained" if language == "English" else "حول البيانات التي تم تدريب النظام عليها"
    if st.button(about_text):
        st.write("Dataset information goes here." if language == "English" else "معلومات حول البيانات المستخدمة هنا.")

    # Load models function
    @st.cache_resource
    def load_models():
        logging.info("Loading models...")
        lstm_model = load_model('model.h5')
        word2vec_model = Word2Vec.load("word2vec_model.bin")
        logging.info("Models loaded successfully.")
        return lstm_model, word2vec_model

    # User Input function with bilingual support
    def get_user_inputs():
        st.markdown("### Please enter your details:" if language == "English" else "### يرجى إدخال التفاصيل الخاصة بك:")

        product_name = st.text_input('Product Name' if language == "English" else "اسم المنتج")
        issue = st.text_area('Issue (in detail)' if language == "English" else "المشكلة (بالتفصيل)")
        company_public_response = st.text_area("Company's Public Response" if language == "English" else "رد الشركة العام")
        company_name = st.text_input("Company Name" if language == "English" else "اسم الشركة")
        tags = st.text_input("Tags (comma separated)" if language == "English" else "العلامات (مفصولة بفواصل)")
        submission_location = st.text_input("Submission Location" if language == "English" else "مكان التقديم")
        company_response = st.text_area("Company's Response" if language == "English" else "رد الشركة")
        timely_response = st.selectbox("Timely Response?" if language == "English" else "استجابة في الوقت المناسب؟", ["Yes", "No"])

        user_inputs = [product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response]
        return user_inputs

    # Text processing
    def process_example_text(example_text, word2vec_model):
        nlp = spacy.blank("en")
        doc = nlp(example_text)
        cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        example_vectors = [word2vec_model.wv[token] for token in cleaned_tokens if token in word2vec_model.wv]
        if not example_vectors:
            example_vectors.append(np.zeros(word2vec_model.vector_size))

        example_vector = np.mean(example_vectors, axis=0)
        return example_vector

    # Prediction function
    def make_prediction(lstm_model, example_vector):
        final_vec = example_vector.reshape((1, example_vector.shape[0], 1))
        prediction = lstm_model.predict(final_vec)[0][0]
        return ("There's a good probability your issue will be disputed." if prediction > 0.5 else "It's quite likely that your issue won't be disputed.")

    # Main function
    def main():
        logging.info("Starting the application...")
        lstm_model, word2vec_model = load_models()

        show_details = st.checkbox("Show Details" if language == "English" else "إظهار التفاصيل")

        if show_details:
            user_inputs = get_user_inputs()
            all_fields_filled = all(input_val.strip() != '' for input_val in user_inputs)

            if all_fields_filled and st.button('Submit' if language == "English" else "إرسال"):
                # Unpack user inputs
                product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response = user_inputs

                # Insert user details into the database
                insert_user_details(product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response)

                st.success("User details saved successfully!" if language == "English" else "تم حفظ تفاصيل المستخدم بنجاح!")

                # Process prediction
                example_vector = process_example_text(issue, word2vec_model)
                prediction = make_prediction(lstm_model, example_vector)
                st.markdown(f"<h2 style='text-align: center; color: #4B0082;'>{prediction}</h2>", unsafe_allow_html=True)
                logging.info(f"Prediction made: {prediction}")
            elif not all_fields_filled:
                st.warning("Please fill in all fields before submitting." if language == "English" else "يرجى ملء جميع الحقول قبل الإرسال.")

if __name__ == '__main__':
    main()

# Close database connection at end of application
conn.close()
