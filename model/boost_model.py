import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns

# Try to import XGBoost with proper error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"XGBoost import error: {e}")
    print("Mac users: Run 'brew install libomp' to install OpenMP runtime required by XGBoost")
    print("Alternatively, try installing xgboost with: pip install xgboost==1.5.0")
    XGBOOST_AVAILABLE = False

# Import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_fetcher import CSVDataFetcher
from indicators import TechnicalIndicators


class BoostModel:
    """XGBoost-based model for stock price prediction using 3-class classification (Sell/Hold/Buy)"""

    def __init__(self, symbol='AAPL', years=3):
        """
        Initialize the model

        Args:
            symbol (str): Stock symbol to train on
            years (int): Number of years of data to use
        """
        self.symbol = symbol
        self.years = years
        self.model = None
        self.indicators = TechnicalIndicators()

        # Check if XGBoost is available
        if not XGBOOST_AVAILABLE:
            print("WARNING: XGBoost not available. Please install it to use this model.")

    def load_data(self):
        """
        Load data for the specified symbol using the data_fetcher

        Returns:
            DataFrame: Processed DataFrame with all indicators
        """
        # Create a data fetcher
        data_fetcher = CSVDataFetcher()

        # Load data
        print(f"Loading data for {self.symbol}...")
        df = data_fetcher.get_data(self.symbol, years=self.years)

        if df is None or df.empty:
            print(f"Error: Could not load data for {self.symbol}")
            return None

        # Add technical indicators
        print("Adding technical indicators...")
        df = self.indicators.add_all_indicators(df)

        # Print summary info
        print(f"Loaded data with shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    def create_labels(self, df):
        """
        Create labels for the dataset based on average return over next 5 days
        Using 3-class classification (Sell/Hold/Buy)

        Args:
            df (DataFrame): DataFrame with price data

        Returns:
            DataFrame: DataFrame with labels added
        """
        # Create a copy to avoid modifying the original
        result = df.copy()

        result['fwd_return'] = result['Close'].shift(-1) / result['Close'] - 1
        result['avg_return_5'] = result['fwd_return'].shift(-1).rolling(window=5).mean() * 100

        # Drop rows with NaN values in avg_return_5
        result = result.dropna(subset=['avg_return_5'])
        print(f"Dropped {len(df) - len(result)} rows with NaN values in avg_return_5")

        # Use thresholds for class separation
        print("\nUsing thresholds for class separation based on 5-day average returns:")
        print("  Sell (0): 5-day avg return < -0.5%")
        print("  Hold (1): -0.5% <= 5-day avg return <= 0.5%")
        print("  Buy (2): 5-day avg return > 0.5%")

        try:
            result['label'] = pd.cut(
                result['avg_return_5'],
                bins=[-np.inf, -0.5, 0.5, np.inf],
                labels=[0, 1, 2]
            )
            result['label'] = result['label'].astype(int)  # Convert from categorical to int
        except ValueError as e:
            print(f"Error in label creation: {e}")
            # If there are issues, use a direct assignment approach
            print("Falling back to direct assignment approach")
            result['label'] = 1  # Default to hold (1)
            result.loc[result['avg_return_5'] < -0.5, 'label'] = 0  # Sell when avg return < -0.5% (0)
            result.loc[result['avg_return_5'] > 0.5, 'label'] = 2   # Buy when avg return > 0.5% (2)


        # Drop rows with missing labels
        result = result.dropna(subset=['label'])

        # Analyze label distribution
        label_counts = result['label'].value_counts().sort_index()
        label_percentages = result['label'].value_counts(normalize=True).sort_index() * 100

        # Print detailed label analysis
        print("\nLabel Distribution Analysis:")
        print(f"Total samples: {len(result)}")

        class_names = ['Sell', 'Hold', 'Buy']
        for i, class_name in enumerate(class_names):
            # Direct mapping: 0 = Sell, 1 = Hold, 2 = Buy
            count = label_counts.get(i, 0)
            percentage = label_percentages.get(i, 0)
            print(f"{class_name} labels ({i}): {count} ({percentage:.2f}%)")

        # Calculate imbalance ratio (max to min)
        if len(label_counts) > 1:
            imbalance_ratio = label_counts.max() / label_counts.min()
            print(f"Imbalance ratio (max:min): {imbalance_ratio:.2f}:1")

            if imbalance_ratio > 2:
                print("WARNING: Class imbalance detected. Consider using class weights or sampling techniques.")

        # Analyze return distribution by class
        print("\nReturn distribution by class:")
        for i, class_name in enumerate(class_names):
            class_returns = result.loc[result['label'] == i, 'avg_return_5']
            if len(class_returns) > 0:
                print(f"  {class_name} ({i}): mean={class_returns.mean():.2f}%, min={class_returns.min():.2f}%, max={class_returns.max():.2f}%")

        # Drop temporary columns
        result = result.drop(['avg_return_5', 'fwd_return'], axis=1)
        return result

    def split_data(self, df, train_size=0.6, val_size=0.2):
        """
        Split the data into training, validation, and test sets using chronological order
        with standard 60/20/20 split

        Args:
            df (DataFrame): DataFrame with features and labels
            train_size (float): Proportion of data for training (default: 0.6 or 60%)
            val_size (float): Proportion of data for validation (default: 0.2 or 20%)

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Sort data by date to ensure chronological order
        df = df.sort_index()

        # Check the distribution of classes
        class_counts = df['label'].value_counts()
        print("\nOverall class distribution before splitting:")
        total_samples = len(df)
        for i, class_name in enumerate(['Sell', 'Hold', 'Buy']):
            count = class_counts.get(i, 0)
            percentage = count / total_samples * 100 if total_samples > 0 else 0
            print(f"  {class_name} ({i}): {count} samples ({percentage:.2f}%)")

        # Calculate split indices
        train_end = int(len(df) * train_size)
        val_end = int(len(df) * (train_size + val_size))

        # Split the data chronologically
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        # Prepare feature columns (excluding labels and other non-feature columns)
        feature_cols = [col for col in df.columns if col not in ['label', 'signal', 'position', 'market_return', 'strategy_return', 'cum_market_return', 'cum_strategy_return']]

        # Create X and y for each set
        X_train = train_df[feature_cols]
        y_train = train_df['label']

        X_val = val_df[feature_cols]
        y_val = val_df['label']

        X_test = test_df[feature_cols]
        y_test = test_df['label']

        # Print summary info
        print("\nData split summary:")
        print(f"Training set: {len(X_train)} samples ({len(X_train)/total_samples*100:.1f}%), from {train_df.index.min()} to {train_df.index.max()}")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/total_samples*100:.1f}%), from {val_df.index.min()} to {val_df.index.max()}")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/total_samples*100:.1f}%), from {test_df.index.min()} to {test_df.index.max()}")

        # Analyze label distribution for each split
        print("\nLabel distribution in splits:")

        # Function to analyze and print distribution for a split
        def print_split_distribution(name, y):
            counts = y.value_counts(normalize=False)
            percentages = y.value_counts(normalize=True) * 100
            print(f"\n{name} set distribution:")
            for i, class_name in enumerate(['Sell', 'Hold', 'Buy']):
                count = counts.get(i, 0)
                percentage = percentages.get(i, 0) if i in percentages else 0
                print(f"  {class_name} ({i}): {count} samples ({percentage:.2f}%)")
            return counts

        train_counts = print_split_distribution("Training", y_train)
        val_counts = print_split_distribution("Validation", y_val)
        test_counts = print_split_distribution("Test", y_test)

        # Calculate distribution shift between splits
        print("\nDistribution shift analysis:")
        all_classes = set(train_counts.index) | set(val_counts.index) | set(test_counts.index)

        for i in sorted(all_classes):
            if i in [0, 1, 2]:  # Only analyze the 3 label classes
                class_name = ['Sell', 'Hold', 'Buy'][i]
                train_pct = train_counts.get(i, 0) / len(y_train) * 100 if len(y_train) > 0 else 0
                val_pct = val_counts.get(i, 0) / len(y_val) * 100 if len(y_val) > 0 else 0
                test_pct = test_counts.get(i, 0) / len(y_test) * 100 if len(y_test) > 0 else 0

                max_shift = max(abs(train_pct - val_pct), abs(train_pct - test_pct), abs(val_pct - test_pct))
                print(f"  {class_name} ({i}): Train {train_pct:.2f}%, Val {val_pct:.2f}%, Test {test_pct:.2f}%, Max shift {max_shift:.2f}%")

                if max_shift > 10:  # More than 10% difference
                    print(f"    WARNING: Significant distribution shift detected for '{class_name}' class")

        # Calculate imbalance ratio for each split
        def calc_imbalance(counts):
            if len(counts) > 1 and counts.min() > 0:
                return counts.max() / counts.min()
            return float('inf')

        train_imbalance = calc_imbalance(train_counts)
        val_imbalance = calc_imbalance(val_counts)
        test_imbalance = calc_imbalance(test_counts)

        print(f"\nImbalance ratios (max:min):")
        print(f"  Training set: {train_imbalance:.2f}:1")
        print(f"  Validation set: {val_imbalance:.2f}:1")
        print(f"  Test set: {test_imbalance:.2f}:1")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the XGBoost model for 3-class classification with enhanced configuration
        to address the issue of poor performance on the Buy class.

        Args:
            X_train (DataFrame): Training features
            y_train (Series): Training labels
            X_val (DataFrame, optional): Validation features
            y_val (Series, optional): Validation labels

        Returns:
            object: Trained model
        """
        if not XGBOOST_AVAILABLE:
            print("ERROR: XGBoost is not available. Cannot train model.")
            return None

        # Analyze class distribution with the medium thresholds
        class_counts = y_train.value_counts().sort_index()
        total_samples = len(y_train)

        # Check if classes are balanced with the new thresholds
        class_percentages = (class_counts / total_samples * 100).to_dict()
        print("\nClass distribution with -0.5/+0.5 thresholds:")
        min_class = class_counts.min()
        max_class = class_counts.max()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

        for class_idx, count in class_counts.items():
            class_name = ['Sell', 'Hold', 'Buy'][class_idx]
            percentage = class_percentages[class_idx]
            print(f"  {class_name} ({class_idx}): {count} samples ({percentage:.2f}%)")

        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

        # Use a more even class weighting approach with power scaling
        print("Using aggressive balancing for Sell and Buy classes")

        # Set explicit weights to strongly favor Sell and Buy classes
        # Sell = 0, Hold = 1, Buy = 2
        class_weights = {
            0: 2.0,  # Strong boost for Sell class
            1: 0.5,  # Reduce weight for Hold class
            2: 2.0   # Strong boost for Buy class
        }

        # Make sure all classes have weights
        for class_idx in class_counts.index:
            if class_idx not in class_weights:
                class_weights[class_idx] = 1.0

        print(f"Applied strong boosting to Sell and Buy classes, reduced Hold class weight")

        # Normalize weights so they sum to number of classes
        weight_sum = sum(class_weights.values())
        class_weights = {k: v * len(class_weights) / weight_sum for k, v in class_weights.items()}

        print("\nFinal class weights:")
        for class_idx, weight in class_weights.items():
            class_name = ['Sell', 'Hold', 'Buy'][class_idx]
            print(f"  {class_name} ({class_idx}): {weight:.4f}")

        # Convert class weights to sample weights for XGBoost
        sample_weights = y_train.map(class_weights)

        # Set parameters for XGBoost with configuration focused on minority class performance
        params = {
            'eta': 0.01,                  # Moderate learning rate
            'max_depth': 3,               # Slightly reduced depth to prevent overfitting to majority class
            'subsample': 0.7,            # More data points to avoid missing minority classes
            'colsample_bytree': 0.7,     # More features per tree for better representation
            'colsample_bylevel': 0.7,    # More features per level
            'min_child_weight': 3,        # Reduced to allow smaller leaf nodes for minority classes
            'alpha': 0.5,                 # Reduced L1 regularization
            'lambda': 1.2,                # Reduced L2 regularization
            'gamma': 0.2,                # Lower min loss reduction to capture more minority patterns
            'max_delta_step': 2,          # Limit step size for more stable learning
            'scale_pos_weight': 1,        # Handle class imbalance
            'tree_method': 'auto',
            'seed': 42,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': ['mlogloss', 'merror'],
            # Class-specific penalties (favors Sell and Buy)
            'base_score': 0.33,           # Equal initial probability for each class
        }

        print("\nTraining parameters optimized for minority class performance:")
        for param, value in params.items():
            if param not in ['tree_method', 'seed', 'objective', 'num_class', 'eval_metric', 'base_score']:
                print(f"  {param}: {value}")

        # Create DMatrix objects for XGBoost with sample weights
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)

        # Create validation set if provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'validation'))

        # Create a proper XGBoost callback class instead of a function
        class ClassDistributionCallback(xgb.callback.TrainingCallback):
            def __init__(self):
                self.iter = 0
                self.best_model = None
                self.best_score = float('inf')
                self.best_model_iter = 0
                self.target_distribution = {0: 0.25, 1: 0.50, 2: 0.25}  # Ideal class balance

            def after_iteration(self, model, epoch, evals_log):
                # Only evaluate every 10 iterations to reduce output
                if self.iter % 10 == 0 and X_val is not None and y_val is not None:
                    # Get current model predictions
                    y_pred_proba = model.predict(dval)
                    y_pred = np.argmax(y_pred_proba, axis=1)

                    # Calculate class distribution
                    class_counts = np.bincount(y_pred, minlength=3)
                    class_distribution = class_counts / len(y_pred)

                    # Calculate validation metrics
                    val_accuracy = np.mean(y_pred == y_val)

                    # Per-class accuracy (balanced metrics)
                    class_accuracy = np.zeros(3)
                    for i in range(3):
                        true_positives = np.sum((y_val == i) & (y_pred == i))
                        class_total = np.sum(y_val == i)
                        class_accuracy[i] = true_positives / class_total if class_total > 0 else 0

                    # Calculate balanced accuracy (average per-class accuracy)
                    balanced_accuracy = np.mean(class_accuracy)

                    # Penalize heavily if not predicting all classes
                    all_classes_penalty = 1.0 if np.all(class_counts > 0) else 5.0

                    # Calculate distribution similarity to target (lower is better)
                    dist_error = 0
                    for i in range(3):
                        # Extra penalty for under-predicting Sell and Buy
                        if i in [0, 2]:  # Sell or Buy
                            dist_error += abs(class_distribution[i] - self.target_distribution[i]) * 2
                        else:
                            dist_error += abs(class_distribution[i] - self.target_distribution[i])

                    # Combine metrics: prioritize balanced accuracy and distribution balance
                    # Lower is better
                    score = (1 - balanced_accuracy) * 3 + dist_error + all_classes_penalty * (1 - np.min(class_accuracy))

                    # Save model if it's better
                    if score < self.best_score:
                        self.best_score = score
                        self.best_model_iter = self.iter
                        self.best_model = model.copy()
                        print(f"Found better model at iteration {self.iter} (score: {score:.4f}, balanced acc: {balanced_accuracy:.4f})")

                    class_percentages = class_counts / len(y_pred) * 100

                    # Print information about class distribution
                    print(f"\n[Iteration {self.iter}] Class prediction distribution:")
                    for i, (count, pct) in enumerate(zip(class_counts, class_percentages)):
                        class_name = ['Sell', 'Hold', 'Buy'][i]
                        acc = class_accuracy[i] * 100
                        print(f"  {class_name} ({i}): {count} samples ({pct:.2f}%), accuracy: {acc:.2f}%")
                    print(f"  Balanced accuracy: {balanced_accuracy:.4f}, Score: {score:.4f}")

                self.iter += 1
                return False  # Continue training

            def after_training(self, model):
                # Always use the best model if found
                if self.best_model is not None:
                    print(f"\nUsing best model from iteration {self.best_model_iter} with score {self.best_score:.4f}")
                    return self.best_model
                return model

            # Other required methods with default implementations
            def before_training(self, model):
                return model

            def before_iteration(self, model, epoch, evals_log):
                return False

        # Train model with more iterations and patience
        print("\nTraining XGBoost model with focus on balanced class performance...")
        callback = ClassDistributionCallback()
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1500,         # Increased iterations for thorough learning
            evals=evals,
            early_stopping_rounds=100,    # More patience for convergence
            verbose_eval=10,              # Print every 10 rounds
            callbacks=[callback]
        )

        print("\nTraining complete!")

        # Get feature importance to understand what the model is using for predictions
        importance = self.model.get_score(importance_type='weight')

        # Print feature usage to diagnose if model is using appropriate features
        print("\nFeature usage in the model:")
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in top_features:
            print(f"  {feature}: {score}")

        # Test the model's prediction distribution on training data
        dtrain_pred = xgb.DMatrix(X_train)
        y_train_pred_proba = self.model.predict(dtrain_pred)
        y_train_pred = np.argmax(y_train_pred_proba, axis=1)

        # Check class distribution in training predictions
        train_pred_counts = np.bincount(y_train_pred, minlength=3)
        train_pred_pct = train_pred_counts / len(y_train_pred) * 100

        print("\nTraining set prediction distribution:")
        for i, (count, pct) in enumerate(zip(train_pred_counts, train_pred_pct)):
            class_name = ['Sell', 'Hold', 'Buy'][i]
            print(f"  {class_name} ({i}): {count} samples ({pct:.2f}%)")

        # Compare with actual distribution
        train_actual_counts = np.bincount(y_train.values, minlength=3)
        train_actual_pct = train_actual_counts / len(y_train) * 100

        print("\nTraining set actual distribution:")
        for i, (count, pct) in enumerate(zip(train_actual_counts, train_actual_pct)):
            class_name = ['Sell', 'Hold', 'Buy'][i]
            print(f"  {class_name} ({i}): {count} samples ({pct:.2f}%)")

        # Check if we're not predicting the Buy class
        if train_pred_counts[2] == 0:
            print("\nWARNING: Model is not predicting Buy class on training data")
            # Check probability distribution for Buy class
            buy_indices = np.where(y_train.values == 2)[0]
            if len(buy_indices) > 0:
                buy_probs = y_train_pred_proba[buy_indices, 2]
                print(f"Average Buy class probability: {np.mean(buy_probs):.4f}")
                print(f"Max Buy class probability: {np.max(buy_probs):.4f}")
                print(f"Samples where Buy probability > 0.2: {np.sum(buy_probs > 0.2)}")

        return self.model

    def validate(self, X_val, y_val):
        """
        Validate the model using validation data

        Args:
            X_val (DataFrame): Validation features
            y_val (Series): Validation labels

        Returns:
            dict: Validation metrics
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            print("ERROR: Model not available. Run train() first.")
            return None

        # Verify the distribution of labels before prediction
        unique_labels = np.unique(y_val)
        print(f"\nValidation labels distribution check:")
        print(f"Unique labels: {unique_labels}")
        value_counts = y_val.value_counts().sort_index()
        for label, count in value_counts.items():
            class_name = ['Sell', 'Hold', 'Buy'][label]
            print(f"  {class_name} ({label}): {count} samples ({count/len(y_val)*100:.2f}%)")

        # Create DMatrix for prediction
        dval = xgb.DMatrix(X_val)

        # Make predictions
        # Multi-class: returns matrix of shape (n_samples, n_classes)
        y_pred_proba = self.model.predict(dval)

        # Verify the shape of prediction probabilities
        print(f"\nPrediction probabilities shape: {y_pred_proba.shape}")
        print(f"Expected classes: 3")

        # Get the predicted class (0, 1, or 2 for sell, hold, buy)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Check distribution of predictions
        pred_unique = np.unique(y_pred)
        print(f"Unique predictions: {pred_unique}")
        for label in range(3):
            class_name = ['Sell', 'Hold', 'Buy'][label]
            count = np.sum(y_pred == label)
            print(f"  Predicted {class_name} ({label}): {count} samples ({count/len(y_pred)*100:.2f}%)")

        # Get classification report with the labels
        print("\nValidation Results:")
        print(classification_report(y_val, y_pred,
                                  target_names=['Sell (0)', 'Hold (1)', 'Buy (2)'],
                                  zero_division=0))

        # Get metrics
        metrics = {}

        # For multi-class, use weighted metrics
        metrics['accuracy'] = accuracy_score(y_val, y_pred)
        metrics['precision'] = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        # Also add macro-averaged metrics
        metrics['precision_macro'] = precision_score(y_val, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_val, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_val, y_pred, average='macro', zero_division=0)

        return metrics

    def test(self, X_test, y_test):
        """
        Test the model using test data with enhanced diagnostics
        for understanding class prediction issues

        Args:
            X_test (DataFrame): Test features
            y_test (Series): Test labels

        Returns:
            dict: Test metrics
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            print("ERROR: Model not available. Run train() first.")
            return None

        # Count samples by class
        class_counts = y_test.value_counts().sort_index()
        print(f"\nTest set class distribution (-0.5/+0.5 thresholds):")
        for i, class_name in enumerate(['Sell', 'Hold', 'Buy']):
            count = class_counts.get(i, 0)
            percent = count / len(y_test) * 100 if len(y_test) > 0 else 0
            print(f"  {class_name} ({i}): {count} samples ({percent:.2f}%)")

        # Check if we have enough samples for each class
        min_class_count = class_counts.min() if not class_counts.empty else 0
        if min_class_count < 10:
            print(f"\nWARNING: Very few samples ({min_class_count}) for minority class in test set")
            print("Performance metrics for minority classes may not be reliable")

        # Calculate class imbalance
        imbalance_ratio = class_counts.max() / class_counts.min() if not class_counts.empty and class_counts.min() > 0 else float('inf')
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 5:
            print("SEVERE class imbalance detected in test set!")
            print("Consider using stratified sampling or different time periods for evaluation")

        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X_test)

        # Make predictions
        # Multi-class: returns matrix of shape (n_samples, n_classes)
        y_pred_proba = self.model.predict(dtest)

        # Verify the shape of prediction probabilities
        print(f"\nPrediction probabilities shape: {y_pred_proba.shape}")

        # Examine the raw prediction probabilities
        print("\nPrediction probability analysis:")
        class_prob_mean = np.mean(y_pred_proba, axis=0)
        class_prob_max = np.max(y_pred_proba, axis=1)
        class_prob_std = np.std(y_pred_proba, axis=0)

        for i, class_name in enumerate(['Sell', 'Hold', 'Buy']):
            print(f"  {class_name} ({i}): mean={class_prob_mean[i]:.4f}, std={class_prob_std[i]:.4f}")

        # Check probability distribution by true class
        print("\nProbability distribution by true class:")
        for true_class in sorted(np.unique(y_test)):
            class_name = ['Sell', 'Hold', 'Buy'][true_class]
            indices = np.where(y_test == true_class)[0]
            if len(indices) > 0:
                class_probs = y_pred_proba[indices, true_class]
                print(f"  True {class_name} ({true_class}) samples:")
                print(f"    Avg probability for correct class: {np.mean(class_probs):.4f}")
                print(f"    Max probability for correct class: {np.max(class_probs):.4f}")
                print(f"    Min probability for correct class: {np.min(class_probs):.4f}")

                # Check if we're struggling with this class
                if np.mean(class_probs) < 0.33:  # Below random guessing
                    print(f"    WARNING: Model is struggling to identify {class_name} class")

        # Get the predicted class (0, 1, or 2 for sell, hold, buy)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Check predicted class distribution
        pred_counts = np.bincount(y_pred, minlength=3)
        print("\nPredicted class distribution:")
        for i, class_name in enumerate(['Sell', 'Hold', 'Buy']):
            count = pred_counts[i]
            percent = count / len(y_pred) * 100
            print(f"  {class_name} ({i}): {count} samples ({percent:.2f}%)")

        # Compare actual vs predicted distributions
        print("\nActual vs Predicted class distribution:")
        actual_dist = np.bincount(y_test.astype(int), minlength=3) / len(y_test)
        pred_dist = pred_counts / len(y_pred)

        for i, class_name in enumerate(['Sell', 'Hold', 'Buy']):
            actual_pct = actual_dist[i] * 100
            pred_pct = pred_dist[i] * 100
            diff = pred_pct - actual_pct
            print(f"  {class_name} ({i}): Actual {actual_pct:.2f}%, Predicted {pred_pct:.2f}%, Diff {diff:+.2f}%")

        # Check for any missing class predictions
        missing_preds = set(range(3)) - set(np.unique(y_pred))
        if missing_preds:
            print(f"\nWARNING: Some classes are never predicted: {[['Sell', 'Hold', 'Buy'][i] for i in missing_preds]}")

            # For each missing class, analyze why it might not be predicted
            for missing_class in missing_preds:
                class_name = ['Sell', 'Hold', 'Buy'][missing_class]
                # Find samples that should belong to this class
                true_samples = np.where(y_test == missing_class)[0]
                if len(true_samples) > 0:
                    # Get their probabilities
                    true_probs = y_pred_proba[true_samples]
                    # Get the probability assigned to the correct class
                    correct_probs = true_probs[:, missing_class]
                    # Get the class that was predicted instead
                    pred_classes = y_pred[true_samples]

                    print(f"\nAnalysis of {len(true_samples)} true {class_name} samples:")
                    predicted_counts = np.bincount(pred_classes, minlength=3)
                    for i, name in enumerate(['Sell', 'Hold', 'Buy']):
                        if i != missing_class and predicted_counts[i] > 0:
                            print(f"  Predicted as {name} ({i}): {predicted_counts[i]} samples")

                    print(f"  Average probability for {class_name}: {np.mean(correct_probs):.4f}")
                    print(f"  Max probability for {class_name}: {np.max(correct_probs):.4f}")

                    # Calculate how close these samples were to being correctly classified
                    winning_probs = np.array([true_probs[j, pred_classes[j]] for j in range(len(true_samples))])
                    margins = winning_probs - correct_probs
                    print(f"  Average margin to correct prediction: {np.mean(margins):.4f}")
                    close_calls = np.sum(margins < 0.05)
                    print(f"  Samples within 0.05 of being correctly predicted: {close_calls} ({close_calls/len(true_samples)*100:.2f}%)")
                else:
                    print(f"\nNo true {class_name} samples in test set to analyze.")

        # Print detailed classification report
        print("\nTest Results:")
        print(classification_report(y_test, y_pred,
                                  target_names=['Sell (0)', 'Hold (1)', 'Buy (2)'],
                                  zero_division=0))

        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"               Predicted")
        print(f"               Sell(0) Hold(1) Buy(2)")
        print(f"Actual Sell(0) {cm[0,0]:6d} {cm[0,1]:6d} {cm[0,2]:6d}")
        print(f"      Hold(1) {cm[1,0]:6d} {cm[1,1]:6d} {cm[1,2]:6d}")
        print(f"      Buy(2)  {cm[2,0]:6d} {cm[2,1]:6d} {cm[2,2]:6d}")

        # Try to visualize the confusion matrix
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))

            # Plot confusion matrix as a heatmap
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                       xticklabels=['Sell', 'Hold', 'Buy'],
                       yticklabels=['Sell', 'Hold', 'Buy'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')

            # Save the plot
            plt.savefig('confusion_matrix.png')
            print("Saved confusion matrix visualization to 'confusion_matrix.png'")
        except Exception as e:
            print(f"Could not create confusion matrix visualization: {e}")

        # Calculate per-class accuracy
        class_accuracy = np.zeros(3)
        for i in range(3):
            if i in np.unique(y_test) and np.sum(y_test == i) > 0:
                class_accuracy[i] = cm[i,i] / np.sum(cm[i,:]) if np.sum(cm[i,:]) > 0 else 0
            else:
                class_accuracy[i] = np.nan  # Use NaN for classes with no samples

        print("\nPer-class accuracy:")
        for i, acc in enumerate(class_accuracy):
            class_name = ['Sell', 'Hold', 'Buy'][i]
            if not np.isnan(acc):
                print(f"  {class_name} ({i}): {acc:.4f}")
            else:
                print(f"  {class_name} ({i}): N/A (no samples)")

        # Detailed analysis of Sell and Buy classes
        print("\nDetailed analysis of minority classes:")

        # Sell class (0) analysis
        sell_indices = np.where(y_test == 0)[0]
        if len(sell_indices) > 0:
            sell_preds = y_pred[sell_indices]
            sell_acc = np.mean(sell_preds == 0)
            sell_confusion = np.bincount(sell_preds, minlength=3)
            print(f"  Sell class ({len(sell_indices)} samples):")
            print(f"    Accuracy: {sell_acc:.4f}")
            print(f"    Predicted as: Sell: {sell_confusion[0]} ({sell_confusion[0]/len(sell_indices)*100:.1f}%), " +
                  f"Hold: {sell_confusion[1]} ({sell_confusion[1]/len(sell_indices)*100:.1f}%), " +
                  f"Buy: {sell_confusion[2]} ({sell_confusion[2]/len(sell_indices)*100:.1f}%)")

            # Analysis of probabilities for Sell class
            sell_probs = y_pred_proba[sell_indices]
            avg_probs = np.mean(sell_probs, axis=0)
            print(f"    Average probabilities: Sell: {avg_probs[0]:.4f}, Hold: {avg_probs[1]:.4f}, Buy: {avg_probs[2]:.4f}")
            print(f"    Probability margin: {avg_probs[0] - np.max(avg_probs[1:]):.4f}")

        # Buy class (2) analysis
        buy_indices = np.where(y_test == 2)[0]
        if len(buy_indices) > 0:
            buy_preds = y_pred[buy_indices]
            buy_acc = np.mean(buy_preds == 2)
            buy_confusion = np.bincount(buy_preds, minlength=3)
            print(f"  Buy class ({len(buy_indices)} samples):")
            print(f"    Accuracy: {buy_acc:.4f}")
            print(f"    Predicted as: Sell: {buy_confusion[0]} ({buy_confusion[0]/len(buy_indices)*100:.1f}%), " +
                  f"Hold: {buy_confusion[1]} ({buy_confusion[1]/len(buy_indices)*100:.1f}%), " +
                  f"Buy: {buy_confusion[2]} ({buy_confusion[2]/len(buy_indices)*100:.1f}%)")

            # Analysis of probabilities for Buy class
            buy_probs = y_pred_proba[buy_indices]
            avg_probs = np.mean(buy_probs, axis=0)
            print(f"    Average probabilities: Sell: {avg_probs[0]:.4f}, Hold: {avg_probs[1]:.4f}, Buy: {avg_probs[2]:.4f}")
            print(f"    Probability margin: {avg_probs[2] - np.max(avg_probs[:2]):.4f}")

        # Calculate balanced accuracy (average of per-class accuracies)
        valid_accuracies = class_accuracy[~np.isnan(class_accuracy)]
        balanced_acc = np.mean(valid_accuracies) if len(valid_accuracies) > 0 else 0
        print(f"Balanced accuracy (mean of per-class accuracies): {balanced_acc:.4f}")

        # Calculate metrics
        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(y_test, y_pred)

        # Use weighted metrics for multi-class
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Add macro-averaged metrics
        metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Add per-class metrics
        for i, acc in enumerate(class_accuracy):
            class_name = ['Sell', 'Hold', 'Buy'][i]
            if not np.isnan(acc):
                metrics[f'accuracy_{class_name}'] = acc

        metrics['balanced_accuracy'] = balanced_acc

        # Calculate performance on minority classes specifically
        minority_classes = []
        for i in range(3):
            if i in class_counts.index and class_counts[i] < len(y_test) / 6:  # Less than 1/6 of data
                minority_classes.append(i)

        if minority_classes:
            print(f"\nMinority class performance (classes {[['Sell', 'Hold', 'Buy'][i] for i in minority_classes]}):")
            # Calculate metrics just for minority classes
            minority_indices = np.isin(y_test, minority_classes)
            if np.any(minority_indices):
                minority_y_test = y_test[minority_indices]
                minority_y_pred = y_pred[minority_indices]

                try:
                    minority_acc = accuracy_score(minority_y_test, minority_y_pred)
                    minority_recall = recall_score(minority_y_test, minority_y_pred, average='macro', zero_division=0)
                    print(f"  Accuracy: {minority_acc:.4f}")
                    print(f"  Recall: {minority_recall:.4f}")

                    metrics['minority_accuracy'] = minority_acc
                    metrics['minority_recall'] = minority_recall
                except Exception as e:
                    print(f"  Could not calculate minority metrics: {e}")

        # Baseline comparison
        # For multi-class: always predict mode
        if not class_counts.empty:
            majority_class = class_counts.idxmax()
            baseline_accuracy = (y_test == majority_class).mean()

            # Map the majority class to its label
            majority_class_name = ['Sell', 'Hold', 'Buy'][majority_class]
            print(f"\nBaseline Accuracy (always predict {majority_class_name}({majority_class})): {baseline_accuracy:.4f}")
            print(f"Model improvement over baseline: {(metrics['accuracy'] - baseline_accuracy) * 100:.2f}%")

            # For balanced accuracy, the baseline would be 1/n_classes if classes are balanced
            n_valid_classes = sum(1 for c in np.unique(y_test))
            balanced_baseline = 1/n_valid_classes if n_valid_classes > 0 else 0
            print(f"Balanced accuracy baseline (random guessing): {balanced_baseline:.4f}")
            print(f"Balanced accuracy improvement: {(balanced_acc - balanced_baseline) * 100:.2f}%")

            metrics['baseline_accuracy'] = baseline_accuracy
            metrics['balanced_baseline'] = balanced_baseline

        # Summary of model performance
        print("\nPerformance Summary:")
        print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Macro-averaged F1: {metrics['f1_macro']:.4f}")

        # Give an overall assessment
        if metrics['balanced_accuracy'] > 0.5:
            print("\nThe model shows meaningful predictive power across all classes.")
        elif metrics['balanced_accuracy'] > 0.33:
            print("\nThe model performs better than random guessing, but still struggles with some classes.")
        else:
            print("\nThe model is not effectively learning patterns for all classes.")

        # Provide improvement suggestions
        print("\nPossible improvements:")
        if imbalance_ratio > 5:
            print("1. Rebalance classes using different thresholds or sampling techniques")
        if metrics['balanced_accuracy'] < 0.4:
            print("2. Explore different features or feature engineering")
        if np.any(class_accuracy < 0.2) and not np.all(np.isnan(class_accuracy)):
            print("3. Investigate specific features that might better predict struggling classes")

        return metrics

    def get_feature_importance(self, feature_names):
        """
        Get feature importance from the model

        Args:
            feature_names (list): List of feature names

        Returns:
            DataFrame: Feature importance
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            print("ERROR: Model not available. Run train() first.")
            return None

        try:
            # Get feature importance scores
            importance = self.model.get_score(importance_type='gain')

            # Check if we got any importance scores
            if not importance:
                print("WARNING: No feature importance scores available.")
                # Create a dummy DataFrame with zero importance
                dummy_data = {'Feature': [], 'Importance': [],
                             'Relative Importance (%)': [], 'Cumulative Importance (%)': []}
                return pd.DataFrame(dummy_data)

            # Create a DataFrame
            importance_df = pd.DataFrame([
                {'Feature': feature, 'Importance': score}
                for feature, score in importance.items()
            ])

            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)

            # Calculate relative importance as percentage
            total_importance = importance_df['Importance'].sum()
            importance_df['Relative Importance (%)'] = (importance_df['Importance'] / total_importance) * 100

            # Calculate cumulative importance
            importance_df['Cumulative Importance (%)'] = importance_df['Relative Importance (%)'].cumsum()

            return importance_df

        except Exception as e:
            print(f"ERROR calculating feature importance: {e}")
            return pd.DataFrame(columns=['Feature', 'Importance',
                                        'Relative Importance (%)', 'Cumulative Importance (%)'])

    def filter_features(self, X_train, X_val=None, X_test=None, importance_threshold=90):
        """
        Filter features based on importance threshold

        Args:
            X_train (DataFrame): Training features
            X_val (DataFrame, optional): Validation features
            X_test (DataFrame, optional): Test features
            importance_threshold (float): Cumulative importance threshold (percentage)

        Returns:
            tuple: Filtered feature DataFrames (X_train_filtered, X_val_filtered, X_test_filtered)
        """
        if not XGBOOST_AVAILABLE or self.model is None:
            print("ERROR: Model not available. Run train() first.")
            return X_train, X_val, X_test

        # Get feature importance
        importance_df = self.get_feature_importance(X_train.columns)

        # Select features up to the importance threshold
        selected_features = importance_df[importance_df['Cumulative Importance (%)'] <= importance_threshold]['Feature'].tolist()

        print(f"\nFeature selection: Keeping {len(selected_features)}/{len(X_train.columns)} features")
        print(f"Selected features account for {importance_threshold:.1f}% of total importance")

        # Filter features
        X_train_filtered = X_train[selected_features]
        X_val_filtered = X_val[selected_features] if X_val is not None else None
        X_test_filtered = X_test[selected_features] if X_test is not None else None

        return X_train_filtered, X_val_filtered, X_test_filtered


if __name__ == "__main__":
    # Check if XGBoost is available
    if not XGBOOST_AVAILABLE:
        print("XGBoost is not available. Please install it to continue.")
        sys.exit(1)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='XGBoost Stock Price Movement Model')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--years', type=int, default=3, help='Years of data to use')
    args = parser.parse_args()

    # Create model instance
    model = BoostModel(symbol=args.symbol, years=args.years)

    # Load data
    df = model.load_data()

    if df is not None:
        # Create multi-class labels (3 classes)
        labeled_df = model.create_labels(df)

        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = model.split_data(labeled_df)

        # Verify label distributions in splits
        print("\nLabel distribution check in train/val/test splits:")
        for name, y in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
            unique_vals = np.unique(y)
            print(f"\n{name} set unique values: {unique_vals}")
            for val in sorted(unique_vals):
                class_name = ['Sell', 'Hold', 'Buy'][val]
                count = (y == val).sum()
                print(f"  Label {class_name} ({val}): {count} samples ({count/len(y)*100:.2f}%)")

        print("\nFeatures used for training:")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Sample features: {', '.join(X_train.columns[:5])}")

        # Train the model with default parameters
        model.train(X_train, y_train, X_val, y_val)

        # Get feature importance for analysis
        importance_df = model.get_feature_importance(X_train.columns)
        if importance_df is not None:
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10)[['Feature', 'Relative Importance (%)', 'Cumulative Importance (%)']])

        # Validate the model
        val_metrics = model.validate(X_val, y_val)

        # Test the model
        test_metrics = model.test(X_test, y_test)

        # Summarize metrics
        print("\nSummary of Performance Metrics:")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        if 'balanced_accuracy' in test_metrics:
            print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")

        # Class-specific metrics
        print("\nClass-specific Metrics:")
        for class_name in ['Sell', 'Hold', 'Buy']:
            if f'accuracy_{class_name}' in test_metrics:
                print(f"  {class_name} Accuracy: {test_metrics[f'accuracy_{class_name}']:.4f}")

        print("\nModel training and evaluation completed successfully!")
