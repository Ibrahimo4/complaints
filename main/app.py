import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
from textblob import TextBlob
import logging
from gensim.models import Word2Vec
import numpy as np
from tensorflow.keras.models import load_model
import spacy

# إعداد قاعدة البيانات
conn = sqlite3.connect('complaints.db', check_same_thread=False)
c = conn.cursor()

# إنشاء جدول الشكاوى
c.execute('''
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        phone TEXT,
        complaint TEXT,
        status TEXT,
        sentiment TEXT,
        timestamp TEXT
    )
''')

# إنشاء جدول تفاصيل المستخدم
c.execute('''
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

# تصميم CSS مخصص
st.markdown(
    """
    <style>
        .stApp { background-color: #F5F5F5; font-family: 'Arial', sans-serif; }
        .sidebar { background-color: #0d47a1; color: white; }
        .sidebar .sidebar-content { background-color: #0d47a1; color: white; }
        .sidebar .sidebar-title { font-size: 24px; font-weight: bold; padding: 10px; text-align: center; }
        .stButton>button { width: 100%; color: white; background-color: #0057A7; margin: 5px 0; border-radius: 5px; padding: 10px; font-weight: bold; }
        .page-title { color: #002B5C; font-weight: bold; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True
)

# الشريط الجانبي للتنقل
st.sidebar.markdown("<div class='sidebar-title'>Complaint Analysis System</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Home", "Submit Complaint", "Complaint Status", "Dashboard", "Manage Complaints", "Using Model", "Reports"])

# Define default language (assuming it's English; you can change this as needed)
language = "English"

def insert_user_details(product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response):
    c.execute('''
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

# إعداد النموذج
@st.cache_resource
def load_models():
    lstm_model = load_model('model.h5')
    word2vec_model = Word2Vec.load("word2vec_model.bin")
    return lstm_model, word2vec_model

# دالة لمعالجة النصوص
def process_example_text(example_text, word2vec_model):
    nlp = spacy.blank("en")
    doc = nlp(example_text)
    cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    example_vectors = [word2vec_model.wv[token] for token in cleaned_tokens if token in word2vec_model.wv]
    if not example_vectors:
        example_vectors.append(np.zeros(word2vec_model.vector_size))
    return np.mean(example_vectors, axis=0)

# دالة التنبؤ
def make_prediction(lstm_model, example_vector):
    final_vec = example_vector.reshape((1, example_vector.shape[0], 1))
    prediction = lstm_model.predict(final_vec)[0][0]
    return "There's a good probability your issue will be disputed." if prediction > 0.5 else "It's quite likely that your issue won't be disputed."

# محتوى الصفحات
if page == "Home":
    st.markdown("<h1 class='page-title'>Welcome to the Customer Complaint Analysis System</h1>", unsafe_allow_html=True)
    st.write("مرحبًا بكم في نظام تحليل شكاوى العملاء.")

elif page == "Submit Complaint":
    st.markdown("<h1 class='page-title'>Submit a Complaint</h1>", unsafe_allow_html=True)
    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    complaint = st.text_area("Complaint Details")

    if st.button("Submit"):
        sentiment_score = TextBlob(complaint).sentiment.polarity
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO complaints (name, email, phone, complaint, status, sentiment, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (name, email, phone, complaint, "Pending", sentiment, timestamp))
        conn.commit()
        st.success("Your complaint has been submitted successfully!")

elif page == "Using Model":
    st.markdown("<h1 class='page-title'>Model</h1>", unsafe_allow_html=True)

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

        return [product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response]

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

elif page == "Complaint Status":
    st.markdown("<h1 class='page-title'>Check Complaint Status</h1>", unsafe_allow_html=True)
    search = st.text_input("Enter your Name or Email to check status")
    if st.button("Search"):
        c.execute("SELECT * FROM complaints WHERE name=? OR email=?", (search, search))
        results = c.fetchall()
        if results:
            for row in results:
                st.write(f"**Complaint ID:** {row[0]}")
                st.write(f"**Status:** {row[5]}")
                st.write(f"**Sentiment:** {row[6]}")
                st.write(f"**Timestamp:** {row[7]}")
                st.write("----")
        else:
            st.warning("No complaints found.")

elif page == "Dashboard":
    st.markdown("<h1 class='page-title'>Complaint Dashboard</h1>", unsafe_allow_html=True)
    complaints_data = pd.read_sql("SELECT * FROM complaints", conn)
    st.subheader("Complaint Overview")
    st.write(f"Total Complaints: {len(complaints_data)}")
    st.write(f"Pending Complaints: {len(complaints_data[complaints_data['status'] == 'Pending'])}")
    st.subheader("Sentiment Analysis")
    sentiment_counts = complaints_data['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
    st.subheader("Complaints Over Time")
    complaints_data['timestamp'] = pd.to_datetime(complaints_data['timestamp'])
    complaints_over_time = complaints_data.groupby(complaints_data['timestamp'].dt.date).size()
    st.line_chart(complaints_over_time)

elif page == "Manage Complaints":
    st.markdown("<h1 class='page-title'>Manage Complaints</h1>", unsafe_allow_html=True)
    complaints_data = pd.read_sql("SELECT * FROM complaints", conn)
    st.write("List of Complaints")
    st.write(complaints_data[['id', 'name', 'complaint', 'status']])
    complaint_id = st.text_input("Enter Complaint ID to update status")
    new_status = st.selectbox("Select New Status", ["Pending", "In Progress", "Resolved"])
    if st.button("Update Status"):
        c.execute("UPDATE complaints SET status=? WHERE id=?", (new_status, complaint_id))
        conn.commit()
        st.success(f"Complaint ID {complaint_id} status updated to {new_status}")

elif page == "Reports":
    st.markdown("<h1 class='page-title'>Generate Reports</h1>", unsafe_allow_html=True)
    report_type = st.selectbox("Select Report Type", ["All Complaints", "Complaints by Sentiment", "Complaints by Status"])
    if report_type == "All Complaints":
        st.write("All Complaints Report")
        complaints_data = pd.read_sql("SELECT * FROM complaints", conn)
        st.write(complaints_data)
    elif report_type == "Complaints by Sentiment":
        st.write("Complaints by Sentiment")
        sentiment_counts = pd.read_sql("SELECT sentiment, COUNT(*) as count FROM complaints GROUP BY sentiment", conn)
        st.bar_chart(sentiment_counts.set_index('sentiment'))
    elif report_type == "Complaints by Status":
        st.write("Complaints by Status")
        status_counts = pd.read_sql("SELECT status, COUNT(*) as count FROM complaints GROUP BY status", conn)
        st.bar_chart(status_counts.set_index('status'))

# إغلاق الاتصال بقاعدة البيانات عند انتهاء التطبيق
conn.close()