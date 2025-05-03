# XGBoost Stock Market Model

This module uses XGBoost to predict stock price movements based on technical indicators.

## Setup Instructions

Before running the model, make sure you have all dependencies installed:

```bash
pip install -r ../requirements.txt
```

### Mac Users

Mac users may encounter issues with XGBoost due to missing OpenMP runtime. To fix this:

1. Install OpenMP using Homebrew:
```bash
brew install libomp
```

2. If you still encounter issues, try installing XGBoost version 1.5.0 specifically:
```bash
pip install xgboost==1.5.0
```

3. If errors persist, try setting these environment variables before running Python:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Running the Model

To train and test the model:

```bash
python boost_model.py
```

This will:
1. Load AAPL stock data for 3 years
2. Add technical indicators
3. Create labels (up/down next day)
4. Split data into train/validation/test sets
5. Train an XGBoost model
6. Validate and test the model
7. Display feature importance

## Alternative Model Types

If XGBoost doesn't work on your system, you can modify the code to use standard scikit-learn models like:
- RandomForestClassifier
- GradientBoostingClassifier

Example modification:
```python
from sklearn.ensemble import GradientBoostingClassifier

# Change the train method to use GradientBoostingClassifier
def train(self, X_train, y_train, X_val=None, y_val=None):
    self.model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    self.model.fit(X_train, y_train)
    return self.model
```

## Troubleshooting

If you see error messages like:
```
XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes:
  * OpenMP runtime is not installed
```

Follow the Mac setup instructions above.