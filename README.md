 # Human Emotions Classification System
Project Overview

This notebook demonstrates the use of machine learning and deep learning techniques for classifying human emotions based on brain signal data. The project uses a LSTM and GRU-based neural network along with several traditional ML classifiers to predict emotional states (Neutral, Sad, Angry) from FNIRS signal data.

📊 Dataset
Source: Book1.csv (7128 rows × 11 columns)

Features: 10 brain signal channels (C1-C10)

Target Variable: Emotion labels (Neutral, Sad, Angry)

Preprocessing:

Label encoding using LabelEncoder

Standard scaling of features

One-hot encoding of labels

Train-test split (80-20)

🧠 Model Architecture

Deep Learning Model (LSTM and GRU-based)
python
Input Layer → GRU(256) → Flatten → Dense(3, softmax)
Total Parameters: 214,275

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Traditional ML Models
Gaussian Naive Bayes (GNB)

Support Vector Machine (SVM)

Logistic Regression (LR)

Decision Tree

Random Forest

📈 Performance Results
LSTM and GRU Model Performance
Training Accuracy: 99.13%

Test Accuracy: 99.13%

Loss on Testing: 2.04

Confusion Matrix (GRU Model)
text
[[422   0   0]
 [  0 462   4]
 [  0   8 483]]
Classification Reports Summary
Gaussian Naive Bayes: 76% accuracy

SVM: 62% accuracy

Logistic Regression: 59% accuracy

Decision Tree: 98% accuracy

Random Forest: 98% accuracy

🔧 Key Functions
Data Transformation (Transform_data())
Encodes labels (Neutral→0, Sad→1, Angry→2)

Scales features using StandardScaler

One-hot encodes labels

Model Creation (create_model())
Creates GRU-based sequential model

Handles time-series nature of brain signals

Outputs 3-class probability distribution

Visualization Functions
Confusion matrix plotting

Model performance visualization

🛠️ Dependencies
python
pandas, numpy, seaborn, matplotlib
scikit-learn, tensorflow/keras
💡 Key Insights
Deep Learning Superiority: The GRU model achieved the highest accuracy (99.13%)

Tree-based Methods: Decision Tree and Random Forest also performed exceptionally well (98%)

Linear Models: Traditional linear classifiers performed moderately (59-76%)

⚠️ Issues Encountered
Label Encoding Error: Initial error with string labels resolved using LabelEncoder

Confusion Matrix Plotting: Attribute error due to column naming mismatch

Library Versions: Required scikit-learn upgrade to version 1.3.0

🚀 How to Use
Ensure all dependencies are installed

Load your dataset in CSV format

Run cells sequentially

Modify model parameters as needed

Evaluate results using provided metrics

📝 Notes
The project demonstrates effective emotion classification from physiological signals

The GRU architecture is particularly suited for sequential/time-series data

Traditional ML models provide good baselines for comparison

Confusion matrices help identify specific emotion classification challenges

🔄 Future Improvements

Add more emotion categories

Implement cross-validation

Add feature importance analysis

Try ensemble methods combining DL and ML approaches
