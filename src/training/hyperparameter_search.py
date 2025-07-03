from sklearn.model_selection import ParameterGrid
import numpy as np
import logging

def hyperparameter_search(model, param_grid, X_train, y_train, scoring_function, n_splits=5):
    best_score = -np.inf
    best_params = None
    
    # Create a grid of hyperparameters
    grid = ParameterGrid(param_grid)
    
    for params in grid:
        logging.info(f"Testing parameters: {params}")
        
        # Set model parameters
        model.set_params(**params)
        
        # Perform cross-validation
        scores = []
        for train_index, val_index in KFold(n_splits=n_splits).split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            model.fit(X_train_fold, y_train_fold)
            score = scoring_function(y_val_fold, model.predict(X_val_fold))
            scores.append(score)
        
        mean_score = np.mean(scores)
        logging.info(f"Mean score for parameters {params}: {mean_score}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            
    logging.info(f"Best parameters found: {best_params} with score: {best_score}")
    return best_params

def main():
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Define model and hyperparameter grid
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Perform hyperparameter search
    best_params = hyperparameter_search(model, param_grid, X, y, accuracy_score)
    print(f"Best hyperparameters: {best_params}")

if __name__ == "__main__":
    main()