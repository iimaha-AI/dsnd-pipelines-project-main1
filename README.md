

# Customer Recommendation Prediction Pipeline

## Project Overview
This project implements a comprehensive machine learning pipeline to predict whether a customer will recommend a product based on their review. The pipeline processes heterogeneous data types (numerical, categorical, and text) using appropriate preprocessing techniques and natural language processing (NLP) methods to build a robust classification model.

## Project Goals
1. Develop an end-to-end machine learning pipeline for text classification
2. Implement appropriate preprocessing techniques for different data types
3. Apply NLP techniques for effective text feature extraction
4. Optimize model performance through hyperparameter tuning
5. Create a reproducible and well-documented workflow

## Dataset Description
The dataset contains customer reviews for clothing products with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| Clothing ID | Integer | Unique identifier for the product |
| Age | Integer | Customer's age |
| Title | Text | Review title |
| Review Text | Text | Full review content |
| Positive Feedback Count | Integer | Number of positive feedbacks received |
| Division Name | Categorical | Product division (e.g., General, General Petite) |
| Department Name | Categorical | Product department (e.g., Dresses, Tops) |
| Class Name | Categorical | Product class (e.g., Dresses, Blouses) |
| Recommended IND | Binary | Target variable (1=Recommended, 0=Not Recommended) |

The dataset contains 23,486 reviews with a class imbalance: approximately 82% positive recommendations and 18% negative recommendations.

## Dependencies
To run this project, you need to install the following packages:

```bash
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
spacy>=3.4.0
notebook>=6.4.0
```

## Installation Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd dsnd-pipelines-project-main
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the spaCy English language model:
```bash
python -m spacy download en_core_web_sm
```

## How to Run the Project

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the project directory and open `starter/starter.ipynb`

3. Execute the notebook cells sequentially. The notebook is organized into the following sections:
   - Load Data: Loads the dataset and displays basic information
   - Explore Data: Performs exploratory data analysis with visualizations
   - Data Preprocessing: Splits data into training and test sets
   - Text Processing: Implements custom text preprocessing using spaCy
   - Building Pipeline: Constructs the ML pipeline with appropriate transformers
   - Training Pipeline: Trains the initial model and evaluates performance
   - Hyperparameter Tuning: Optimizes model parameters using GridSearchCV
   - Evaluating the Model: Assesses the optimized model's performance
   - Feature Importance: Analyzes key features influencing predictions
   - Testing the Model: Demonstrates predictions on new data

## Project Structure
```
dsnd-pipelines-project-main/
├── .gitignore
├── CODEOWNERS
├── LICENSE.txt
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── recommendation_model.pkl  # Trained model (generated after execution)
└── starter/
    ├── README.md             # Starter folder documentation
    ├── data/
    │   └── reviews.csv       # Dataset
    └── starter.ipynb         # Jupyter notebook with implementation
```

## Pipeline Design and Technical Choices

### Text Preprocessing Pipeline
The text processing pipeline uses spaCy for NLP tasks with these specific design choices:

1. **Lowercasing and Special Character Removal**:
   - Converts all text to lowercase to ensure consistency
   - Removes non-alphabetic characters to reduce noise in the data
   - Rationale: Standardizes text input and focuses on meaningful content

2. **Lemmatization**:
   - Uses spaCy's lemmatizer to reduce words to their base dictionary form
   - Example: "running" → "run", "better" → "good"
   - Rationale: More effective than stemming for preserving word meaning while reducing dimensionality

3. **Stop Word Removal**:
   - Eliminates common English stop words (e.g., "the", "is", "at")
   - Rationale: Removes low-information words that don't contribute to sentiment prediction

4. **TF-IDF Vectorization**:
   - Converts processed text into numerical features using Term Frequency-Inverse Document Frequency
   - Uses n-grams (1,2) to capture both single words and meaningful phrases
   - Limits features to prevent overfitting (100 for titles, 500 for reviews)
   - Rationale: Balances feature richness with computational efficiency

### Data Type Handling
The pipeline implements specialized transformers for each data type:

1. **Numerical Features**:
   - Median imputation for missing values
   - Standard scaling for normalization
   - Rationale: Median is robust to outliers; scaling ensures equal feature contribution

2. **Categorical Features**:
   - Most frequent imputation for missing values
   - One-hot encoding for categorical variables
   - Rationale: Preserves categorical information without ordinal assumptions

3. **Text Features**:
   - Custom text processing pipeline
   - TF-IDF vectorization for feature extraction
   - Rationale: Captures semantic meaning in text data

### Model Selection
**Random Forest Classifier** was chosen for several reasons:
- Handles high-dimensional data effectively
- Provides built-in feature importance metrics
- Resistant to overfitting through ensemble averaging
- Performs well with mixed data types
- Requires minimal hyperparameter tuning to achieve good results

### Pipeline Architecture
The pipeline uses scikit-learn's `Pipeline` and `ColumnTransformer` to:
1. Ensure consistent preprocessing between training and inference
2. Prevent data leakage from test set
3. Create a modular, maintainable workflow
4. Enable easy hyperparameter tuning across all components

## Results

### Model Performance Metrics

| Metric | Initial Model | Optimized Model | Improvement |
|--------|---------------|----------------|-------------|
| Accuracy | 0.8620 | 0.8634 | +0.14% |
| Precision | 0.8682 | 0.8688 | +0.06% |
| Recall | 0.9797 | 0.9807 | +0.10% |
| F1 Score | 0.9206 | 0.9214 | +0.08% |

### Classification Report (Optimized Model)
```
              precision    recall  f1-score   support
           0       0.80      0.34      0.48       678
           1       0.87      0.98      0.92      3011
    accuracy                           0.86      3689
   macro avg       0.83      0.66      0.70      3689
weighted avg       0.86      0.86      0.84      3689
```

### Key Observations
1. The model excels at identifying recommended products (class 1) with 98% recall
2. Performance on non-recommended products (class 0) is weaker (34% recall)
3. The class imbalance (82% positive vs 18% negative) affects model performance
4. Hyperparameter tuning provided marginal improvements across all metrics
5. High precision for class 1 (87%) indicates reliable positive predictions

### Sample Predictions
```
Review 1: "I love this dress, it fits perfectly and looks amazing."
Prediction: Recommended (Confidence: 99.00%)

Review 2: "This jumpsuit is so comfortable and stylish."
Prediction: Recommended (Confidence: 98.00%)
```

## Challenges and Solutions

1. **Class Imbalance**:
   - Challenge: Significant imbalance between positive and negative reviews
   - Solution: Used stratified sampling during train-test split to maintain class distribution
   - Impact: Ensures model evaluation reflects real-world class distribution

2. **Text Processing Complexity**:
   - Challenge: Processing raw text while preserving meaningful information
   - Solution: Implemented comprehensive NLP pipeline with lemmatization and stop word removal
   - Impact: Reduced noise while maintaining semantic content

3. **Mixed Data Types**:
   - Challenge: Combining numerical, categorical, and text features in a single pipeline
   - Solution: Used ColumnTransformer to apply appropriate preprocessing to each feature type
   - Impact: Created a unified workflow that handles all data types correctly

4. **Model Optimization**:
   - Challenge: Finding optimal hyperparameters for the Random Forest classifier
   - Solution: Implemented GridSearchCV with 5-fold cross-validation
   - Impact: Systematically explored parameter space to improve model performance

## Future Improvements

1. **Address Class Imbalance**:
   - Implement techniques like SMOTE or ADASYN
   - Use class weights in the classifier
   - Experiment with different evaluation metrics (e.g., AUC-ROC)

2. **Advanced Text Processing**:
   - Experiment with word embeddings (Word2Vec, GloVe)
   - Implement transformer-based models (BERT, DistilBERT)
   - Add sentiment analysis features

3. **Model Enhancement**:
   - Try gradient boosting models (XGBoost, LightGBM)
   - Implement neural network architectures
   - Ensemble multiple models for improved performance

4. **Operational Improvements**:
   - Create a web interface for model predictions
   - Implement model monitoring and retraining pipeline
   - Add comprehensive unit tests

5. **Documentation and Usability**:
   - Create detailed API documentation
   - Add more visualization options
   - Implement interactive dashboard for results exploration

## License
This project is licensed under the terms of the LICENSE.txt file.

## Acknowledgments
This project was completed as part of the Data Scientist Nanodegree program from Udacity. The dataset was provided for educational purposes.
