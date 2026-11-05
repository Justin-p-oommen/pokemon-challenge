import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.base import clone

# logistic regression (want to test different C and penalties)
def get_logistic_regression(C=0.1, penalty='l1'):
    # returns a regularized Logistic Regression pipeline
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty=penalty, solver='liblinear', C=C, max_iter=3000, random_state=42)
    )

# tuned xgb model
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


# combination of lr and xgb model
def get_ensemble_model():
    
    #Combines Logistic Regression and XGBoost using Soft Voting.
    #Soft voting averages the predicted probabilities of both models.
    
    return VotingClassifier(
        estimators=[
            ('lr', get_logistic_regression(C=0.1, penalty='l2')), # used l2
            ('xgb', get_tuned_xgboost())
        ],
        voting='soft', # averages probabilities
        weights=[weight_lr, weight_xgb],
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

    # Train model
    val_model = clone(model)
    val_model.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train,val_model.predict(X_train))
    test_acc = accuracy_score(y_test, val_model.predict(X_test))
    gap = train_acc - test_acc

    print(f"Train Accuracy:  {train_acc * 100:.2f}%")
    print(f"Test Accuracy:   {test_acc * 100:.2f}%")
    print(f"Overfitting Gap: {gap * 100:.2f}%\n")

    #if gap > 0.05:
     #   print(" High Overfitting !!!")
    #elif gap < -0.02:
     #   print(" No Overfitting. Generalizes perfectly.")
        
    # K-fold validation
    print(f"\n--- {n_splits}-Fold RepeatStratified Cross-Validation ---")
    
    #cv = RepeatedStratifiedKFold(n_splits=n_splits, random_state=42)
    #cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    cv_scores = cross_val_score(clone(model), X, y, cv=cv_splits, scoring='accuracy', n_jobs=-1)
    
    #for i, score in enumerate(cv_scores):
     #   print(f"Fold {i+1}: {score * 100:.2f}%")
        
    print(f"\nMean CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # for the whole train.json
    print("\nFitting final model on 100% of the data...")
    model.fit(X, y)
        
    return model

def get_feature_importance(df, target_col='player_won', drop_cols=['battle_id']):
    correlation_matrix = df.corr()
    feature_impact = correlation_matrix[target_col].drop(drop_cols + [target_col]).sort_values(ascending=False)
    return feature_impact
