# plant_disease_prediction
Plant 🌱 Disease 🐛 Detection 🔎
Plant Disease Detection is an innovative machine learning project that harnesses the power of Convolutional Neural Networks (CNN) and deep learning techniques to identify and classify diseases in plants. The primary objective is to offer farmers and agricultural experts a valuable tool for swift plant health diagnosis, facilitating timely intervention and minimizing the risk of crop loss.

Project Structure 📂
The project comprises essential components:

Plant_Disease_Detection.ipynb: Jupyter Notebook with the code for model training.

main_app.py: Streamlit web application for plant disease prediction.

plant_disease_model.h5: Pre-trained model weights.

requirements.txt: List of necessary Python packages.

Train_plant_disease.ipynb: Jupyter Notebook for training the plant disease detection model.

Test_Plant_Disease.ipynb: Jupyter Notebook for testing the plant disease detection model.

Installation 🚀
To run the project locally, follow these steps:

Navigate to the project directory:

cd Plant-Disease-Detection

Install the required packages:

pip install -r requirements.txt

Run the Streamlit web application:

streamlit run main_app.py

Usage 🌿
Once the application is running, open your web browser and navigate to http://localhost:8501. Upload an image of a plant leaf, and the system will predict if it is affected by any disease.

Model Training 🧠
The model was trained using the Train_plant_disease.ipynb notebook. It employs a Convolutional Neural Network architecture to classify plant images into different disease categories. The trained model weights are saved in plant_disease_model.h5.

Web Application 🌐
The web application (main_app.py) empowers users to interact with the trained model. Upload plant images, and the application provides real-time predictions regarding the health of the plant. The application also supports multiple languages, including major Indian languages.

Requirements 🛠️
Keras

Numpy

Streamlit

OpenCV-Python-Headless

TensorFlow

-End-
