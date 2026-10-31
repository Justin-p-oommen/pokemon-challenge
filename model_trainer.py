from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_evaluate(df, target_col='player_won', drop_cols=['battle_id']):
    # Define features and target
    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, accuracy

def get_feature_importance(df, target_col='player_won', drop_cols=['battle_id']):
    correlation_matrix = df.corr()
    feature_impact = correlation_matrix[target_col].drop(drop_cols + [target_col]).sort_values(ascending=False)
    return feature_impact
