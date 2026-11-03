import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


def get_logistic_regression():
    # returns a regularized Logistic Regression pipeline
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000, random_state=42)
    )

def get_random_forest():
    # returns a regularized Random Forest
    return RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=30, random_state=42, n_jobs=-1
    )

# my model
'''
def get_tuned_xgboost():
    # returns the heavily tuned XGBoost model
    return xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=3, min_child_weight=5,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
'''

# tune1
def get_tuned_xgboost():
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

'''
# tune2
def get_tuned_xgboost():
    return xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.5,
        reg_lambda=2.0,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
'''

# combination of lr and xgb model
def get_ensemble_model():
    
    #Combines Logistic Regression and XGBoost using Soft Voting.
    #Soft voting averages the predicted probabilities of both models.
    
    return VotingClassifier(
        estimators=[
            ('lr', get_logistic_regression()),
            ('xgb', get_tuned_xgboost())
        ],
        voting='soft', # Averages probabilities (better than 'hard' voting)
        n_jobs=-1
    )



def train_and_evaluate(df, model, model_name, target_col='player_won', drop_cols=['battle_id'], n_splits=5):

    print(f"\n{'='*45}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*45}")
    
    # Define features and target
    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col]

    print("--- 80/20 Split (Checking Overfitting) ---")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    gap = train_acc - test_acc

    print(f"Train Accuracy:  {train_acc * 100:.2f}%")
    print(f"Test Accuracy:   {test_acc * 100:.2f}%")
    print(f"Overfitting Gap: {gap * 100:.2f}%\n")

    if gap > 0.05:
        print(" High Overfitting !!!")
    elif gap < -0.02:
        print(" No Overfitting. Generalizes perfectly.")
        
    # K-fold validation
    print(f"\n--- {n_splits}-Fold Stratified Cross-Validation ---")
    
    cv = RepeatedStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1}: {score * 100:.2f}%")
        
    print(f"\nMean CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # for the whole train.json
    print("\nFitting final model on 100% of the data...")
    model.fit(X, y)
        
    return model

def get_feature_importance(df, target_col='player_won', drop_cols=['battle_id']):
    correlation_matrix = df.corr()
    feature_impact = correlation_matrix[target_col].drop(drop_cols + [target_col]).sort_values(ascending=False)
    return feature_impact
