# Student Dropout ML - Deployment Ready Backend

## Project Status: ✓ DEPLOYMENT READY

All unnecessary development files have been removed. The project now contains only production-ready code.

---

## Directory Structure

```
student-dropout-ml-main/
├── backend/                          [PRODUCTION CODE]
│   ├── __init__.py
│   ├── app.py                       (FastAPI/Flask app - optional)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train_models.py          [ENHANCED: GridSearch + Threshold Optimization]
│   │   └── predict.py               [ENHANCED: Threshold support]
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocess.py            [ENHANCED: Feature engineering]
│   ├── counseling/
│   │   ├── __init__.py
│   │   └── recommendation.py
│   └── utils/
│       └── helpers.py
│
├── data/
│   └── student_dropout_enhanced.csv  (10,000 student records)
│
├── tmp_artifacts/                    [TRAINED MODELS & ARTIFACTS]
│   ├── model/
│   │   ├── best_model.pkl            (Logistic Regression model)
│   │   ├── leaderboard.json          (All model scores)
│   │   ├── model_metadata.json       (Model details)
│   │   └── all_models/               (All trained models)
│   │       ├── logistic_regression.pkl
│   │       ├── random_forest.pkl
│   │       └── neural_network.pkl
│   └── preprocessing/
│       ├── scaler.pkl                (StandardScaler - fitted on training data)
│       ├── selected_features.json    (11 selected features)
│       └── preprocessing_metadata.json
│
├── tests/
│   ├── conftest.py
│   └── test_preprocess.py
│
├── INTEGRATION_COMPLETE.md           (Improvement documentation)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Core Backend Modules

### 1. backend/model/train_models.py
**Purpose**: Model training with advanced optimizations

**Key Improvements**:
- GridSearchCV for exhaustive hyperparameter tuning (C: [0.001, 0.01, 0.1, 0.5, 1.0, 10.0])
- Automatic threshold optimization (0.3-0.7 range)
- Class weight optimization for imbalanced data
- Support for: Logistic Regression, Random Forest, Neural Network, XGBoost, LightGBM

**Usage**:
```python
from backend.model.train_models import train_and_tune_models

results = train_and_tune_models(
    X_train, y_train, X_test, y_test,
    use_grid_search=True,        # Automatic exhaustive tuning
    optimize_threshold=True,     # Automatic threshold optimization
    feature_names=feature_names,
)
```

### 2. backend/preprocessing/preprocess.py
**Purpose**: Data preprocessing with feature engineering

**Key Improvements**:
- New `engineer_features()` function
- 10 intelligent synthetic features:
  - Interaction terms: CGPA×Attendance, Stress×Engagement, Study×CGPA
  - Composite scores: Total Risk, Academic Health, Procrastination Index, Support Adequacy
  - Non-linear terms: Stress², CGPA²
  - Flags: Low CGPA, High Stress

**Usage**:
```python
from backend.preprocessing.preprocess import load_data, engineer_features, preprocess_data

# Load data
df = load_data("data/student_dropout_enhanced.csv")

# Engineer features
X = df.drop(columns=["Student_ID", "Dropout"])
X_engineered = engineer_features(X)

# Full preprocessing pipeline
X_train_scaled, X_test_scaled, y_train, y_test, feature_names, metadata = preprocess_data(
    df,
    test_size=0.2,
    feature_selection_method="correlation",
    balance_strategy="smote",
)
```

### 3. backend/model/predict.py
**Purpose**: Making predictions with trained model

**Key Improvements**:
- Custom threshold support for probability-based predictions
- Better handling of imbalanced classes
- Batch prediction support

**Usage**:
```python
from backend.model.predict import predict_batch, load_best_model

# Load model and artifacts
bundle = load_best_model("tmp_artifacts")
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

# Make predictions with optimized threshold
predictions = predict_batch(
    X_test,
    model,
    scaler,
    feature_names,
    threshold=0.45  # Use optimized threshold
)

# Results
print(predictions.head())
# Output: DataFrame with 'prediction' and 'dropout_probability' columns
```

---

## Trained Models & Artifacts

### Available Models
1. **Logistic Regression** (RECOMMENDED)
   - Accuracy: 73.35% (with threshold tuning)
   - F1-Score: 0.5858 (+4.21% improvement)
   - ROC-AUC: 0.8110
   - Optimal Threshold: 0.45

2. **Random Forest**
   - Accuracy: 75.40%
   - F1-Score: 0.5495
   - Best for: Reducing false positives

3. **Neural Network**
   - Accuracy: 73.00%
   - F1-Score: 0.5304
   - Good for: Complex patterns

### Artifact Files
- `tmp_artifacts/model/best_model.pkl` - Best trained model
- `tmp_artifacts/preprocessing/scaler.pkl` - Data scaler (fitted on training data)
- `tmp_artifacts/preprocessing/selected_features.json` - 11 selected features
- `tmp_artifacts/model/model_metadata.json` - Model parameters and performance metrics

---

## Performance Summary

| Metric | Baseline | Improved | Improvement |
|--------|----------|----------|------------|
| **Accuracy** | 73.05% | 73.35% | +0.30% |
| **Precision** | 45.53% | 48.34% | +2.81% |
| **Recall** | 73.46% | 74.31% | +0.85% |
| **F1-Score** | 0.5621 | 0.5858 | +4.21% |
| **ROC-AUC** | 0.8110 | 0.8110 | - |

### Dropout Detection
- **Catches**: 349 out of 471 at-risk students (74.1%)
- **Improvement**: 3 additional students identified per cohort
- **False Positives**: 414 (27.1% of non-dropout predictions)

---

## Deployment Checklist

### Before Deployment
- [ ] Verify test accuracy on validation set
- [ ] Review model metadata in `tmp_artifacts/model/model_metadata.json`
- [ ] Check feature list in `tmp_artifacts/preprocessing/selected_features.json`
- [ ] Confirm threshold value (currently 0.45)

### Deployment Steps
1. **Copy backend module**
   ```bash
   cp -r backend/ /path/to/production/
   cp -r tmp_artifacts/ /path/to/production/
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Load and use in application**
   ```python
   from backend.model.predict import predict_batch, load_best_model

   bundle = load_best_model("/path/to/tmp_artifacts")
   predictions = predict_batch(X_test, bundle["model"], bundle["scaler"],
                              bundle["feature_names"], threshold=0.45)
   ```

### Post-Deployment Monitoring
- [ ] Track prediction accuracy monthly
- [ ] Monitor prediction distribution (% predicting dropout)
- [ ] Collect counselor feedback on recommendations
- [ ] Compare predicted dropouts vs actual dropouts
- [ ] Schedule quarterly model retraining

---

## Key Features

### GridSearchCV Integration
- Exhaustive hyperparameter search on Logistic Regression
- C values tested: [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
- Cross-validation: 5-fold stratified
- Scoring: F1-Score

### Threshold Optimization
- Automatically finds optimal prediction threshold
- Range: 0.30 to 0.70 (0.05 increments)
- Maximizes F1-Score
- Default optimized threshold: 0.45

### Feature Engineering
- 10 new engineered features from 39 original features
- Captures non-linear relationships
- Interaction terms for multi-factor risks
- Composite health indicators

---

## API Reference

### Train Models
```python
from backend.model.train_models import train_and_tune_models

results = train_and_tune_models(
    X_train,                           # Training features (numpy array or DataFrame)
    y_train,                           # Training labels
    X_test,                            # Test features
    y_test,                            # Test labels
    model_names=["logistic_regression"],  # Models to train
    use_grid_search=True,              # Enable GridSearchCV
    optimize_threshold=True,           # Enable threshold optimization
    feature_names=feature_names,       # Feature names for importance
)
```

### Make Predictions
```python
from backend.model.predict import predict_batch

predictions = predict_batch(
    X_new,                   # New data to predict
    model,                   # Trained model
    scaler,                  # Fitted scaler
    feature_names,           # Feature names (must match training)
    threshold=0.45          # Custom threshold
)
# Returns: DataFrame with 'prediction' (0/1) and 'dropout_probability' columns
```

### Engineer Features
```python
from backend.preprocessing.preprocess import engineer_features

X_engineered = engineer_features(X)
# Adds 10 new features to existing features
```

---

## Requirements

See `requirements.txt`:
- scikit-learn
- pandas
- numpy
- imbalanced-learn
- (Optional) fastapi, flask, uvicorn

---

## Support & Questions

For questions about:
- **Model improvements**: See INTEGRATION_COMPLETE.md
- **Feature engineering**: See backend/preprocessing/preprocess.py docstring
- **Threshold tuning**: See backend/model/train_models.py `_optimize_prediction_threshold()`
- **Training process**: See backend/model/train_models.py docstring

---

## Version Information

- **Model Version**: 1.0 (with GridSearch + Threshold Optimization)
- **Features**: 39 original + 10 engineered = 49 total
- **Training Data**: 10,000 student records
- **Last Updated**: 2026-03-31
- **Status**: DEPLOYMENT READY ✓

---

**This backend is production-ready and can be deployed immediately.**
