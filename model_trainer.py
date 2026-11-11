"""
model_trainer.py

What is this file?

This file handles the machine learning part of the project. It defines the models, 
trains them, and evaluates how well they predict if Player 1 will win.

The code has two main models:
1. Logistic Regression: A simple linear model that uses L1 or L2 penalty to prevent overfitting.
2. 2. Random Forest: A bagging tree-based model.
3. XGBoost: A powerful tree-based model tuned to be conservative (shallow trees) so it generalizes well.

It also contains the evaluation function. This function does an 80/20 split to check the overfitting gap, and then uses 15-Fold Cross-Validation (5 splits repeated 3 times) to get a reliable accuracy score.

Input: The dataframe of features and the model to test.

Output: The trained model and print statements showing the accuracy scores.
"""

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.base import clone

# logistic regression (want to test different C and penalties)
def get_logistic_regression(C=0.1, penalty='l1'):
    # returns a regularized Logistic Regression pipeline
    # I use StandardScaler because Logistic Regression needs scaled data.
    # C is the inverse of regularization strength (smaller C = stronger regularization).
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty=penalty, solver='liblinear', C=C, max_iter=3000, random_state=42)
    )

# Random Forest Model
def get_random_forest():
    # Returns a regularized Random Forest.
    # I limited the depth (6) and increased the minimum samples per leaf (30) 
    # to prevent the trees from memorizing the training data.
    return RandomForestClassifier(
        n_estimators=300, 
        max_depth=6, 
        min_samples_leaf=30, 
        random_state=42, 
        n_jobs=-1
    )


# tuned xgb model
def get_tuned_xgboost():
    # Returns the heavily tuned XGBoost model.
    # I made the trees very shallow (max_depth=2) and used a slow learning rate (0.01) 
    # with strong L1/L2 penalties (reg_alpha, reg_lambda) so it doesn't memorize the training data.
    return xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=2,
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=2.0,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

def train_and_evaluate(df, model, model_name, cv_splits, target_col='player_won', drop_cols=['battle_id']):

    print(f"\n{'='*45}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*45}")
    
    # Define features and target
    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col]

    print("--- 80/20 Split (Checking Overfitting) ---")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # I clone the model so that the CV loop doesn't mess up the original model 
    # before we do the final fit at the bottom.
    val_model = clone(model)
    val_model.fit(X_train, y_train)

    # Evaluate the 80/20 split
    train_acc = accuracy_score(y_train,val_model.predict(X_train))
    test_acc = accuracy_score(y_test, val_model.predict(X_test))
    gap = train_acc - test_acc

    print(f"Train Accuracy:  {train_acc * 100:.2f}%")
    print(f"Test Accuracy:   {test_acc * 100:.2f}%")
    print(f"Overfitting Gap: {gap * 100:.2f}%\n")
        
    # K-fold validation
    print(f"\n--- RepeatStratified Cross-Validation ---")

    # I use the shared cv_splits so all models are tested on the exact same data splits
    # shared CV Splits
    if cv_splits is None:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        cv_splits = list(cv.split(X, y))

    # I clone the model here again for the CV to keep it clean
    cv_scores = cross_val_score(clone(model), X, y, cv=cv_splits, scoring='accuracy', n_jobs=-1)
    
        
    print(f"\nMean CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    
    # --- Final Fit ---
    # Fit the model on 100% of the data so it learns from all battles before submission
    print("\nFitting final model on 100% of the data...")
    model.fit(X, y)
        
    return model

def get_feature_importance(df, target_col='player_won', drop_cols=['battle_id']):
    # Calculates the Pearson correlation between features and the target.
    # Note: I used Permutation Importance for the final report because correlation 
    # only captures straight-line relationships.
    correlation_matrix = df.corr()
    feature_impact = correlation_matrix[target_col].drop(drop_cols + [target_col]).sort_values(ascending=False)
    return feature_impact
