# Consumer-Complaint-Analysis-AIOPS-PROJECT-

## Introduction
Welcome to the Consumer Complaint Disputer, an AI-driven customer dispute system designed and built by our team. This web application predicts whether a given consumer complaint will be disputed or not.

## About the Project
The purpose of this project is to design and build a scalable machine learning pipeline to predict whether a consumer complaint will be disputed or not.

## Project Details
The application is built using Python and incorporates the following technologies and libraries:
- `numpy`: For numerical operations.
- `spacy`: For natural language processing.
- `gensim`: For Word2Vec model.
- `streamlit`: For building the web interface.
- `tensorflow.keras`: For loading the LSTM model.
- `logging`: For logging application events.
  
## Application Structure
- `app.py`: The main application script containing the Streamlit web interface and model integration.
- `database.py`: Module for database operations (not provided in the code snippet).

## Usage
To use the application, follow these steps:
1. Run the application using Streamlit: `streamlit run app.py`.
2. Fill in the required details such as product name, issue, company response, etc.
3. Click the "Show Prediction" button to get the model's prediction.

## Models
The application uses a LSTM model (`lstm_model.h5`) and a Word2Vec model (`word2vec_model.bin`). These models are responsible for predicting dispute outcomes and processing user inputs.

## Logging
Application logs are stored in the `app.log` file. It contains information about model loading, user inputs, and predictions.

## Project Details
The purpose of this project is to design and build a scalable machine learning pipeline to predict whether a consumer complaint will be disputed or not.

## Team Members
- [Arya Chakraborty](https://www.linkedin.com/in/arya-chakraborty2002/)
- [Rituparno Das](https://www.linkedin.com/in/das-rituparno/)
- [Prathamesh](https://www.linkedin.com/in/prathmesh-hambarde-01050b234/)
- [Bharadwaj]()

## Dependencies
Make sure to install the required dependencies before running the application:
```bash
pip install numpy spacy gensim streamlit tensorflow
