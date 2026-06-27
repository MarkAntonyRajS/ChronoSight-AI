import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error, silhouette_score
)

# Baseline models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans

class AutoMLPipeline:
    def __init__(self):
        self.task_type: Optional[str] = None  # 'classification', 'regression', 'clustering'
        self.best_model: Any = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_cols: List[str] = []
        self.target_col: Optional[str] = None
        self.metrics: Dict[str, float] = {}
        self.model_name: str = ""
        self.feature_importance_df: pd.DataFrame = pd.DataFrame()
        
        # Metadata to help generate prediction UI
        self.categorical_options: Dict[str, List[Any]] = {}
        self.numeric_bounds: Dict[str, Dict[str, float]] = {}
        
    def predict(self, raw_input: Dict[str, Any]) -> Tuple[Any, Optional[np.ndarray]]:
        """
        Takes a raw input dictionary of feature values, preprocesses it,
        and returns the prediction (and class probabilities if classification).
        """
        if not self.best_model or not self.preprocessor:
            raise ValueError("Model pipeline is not trained yet.")
            
        # Convert raw input to DataFrame matching training structure
        input_df = pd.DataFrame([raw_input])
        
        # Ensure all training feature columns exist, fill missing with np.nan
        for col in self.feature_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan
            else:
                # If numeric column but passed as formatting string (e.g. from tests), clean it!
                if col in self.numeric_bounds and (isinstance(input_df[col].iloc[0], str) or input_df[col].dtype == object):
                    val_str = str(input_df[col].iloc[0])
                    cleaned_val = re.sub(r'[$,%\s]', '', val_str).replace(',', '')
                    try:
                        input_df[col] = pd.to_numeric(pd.Series([cleaned_val]), errors='coerce').astype(float)
                    except Exception:
                        pass
                
        # Reorder columns to match feature_cols
        input_df = input_df[self.feature_cols]
        
        # Preprocess
        processed_x = self.preprocessor.transform(input_df)
        
        # Predict
        prediction = self.best_model.predict(processed_x)
        probabilities = None
        
        if self.task_type == "classification" and hasattr(self.best_model, "predict_proba"):
            try:
                probabilities = self.best_model.predict_proba(processed_x)[0]
            except Exception:
                pass
                
        return prediction[0], probabilities

def detect_task_type(df: pd.DataFrame, target_col: Optional[str]) -> str:
    """
    Detects whether the target variable requires Regression or Classification.
    Defaults to Clustering if no target is provided.
    """
    if not target_col:
        return "clustering"
        
    target_series = df[target_col].dropna()
    if len(target_series) == 0:
        raise ValueError(f"Target column '{target_col}' contains only missing values.")
        
    col_dtype = target_series.dtype
    unique_count = target_series.nunique()
    
    if pd.api.types.is_numeric_dtype(col_dtype):
        # Numeric with low cardinality is classification (e.g. binary indicators or classes represented as integers)
        if unique_count <= 10:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"

def get_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Creates a preprocessor pipeline for numeric and categorical columns.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

def run_automl(df: pd.DataFrame, target_col: Optional[str] = None, random_state: int = 42) -> Tuple[AutoMLPipeline, List[str]]:
    """
    Runs an end-to-end AutoML pipeline.
    Identifies task, preprocesses data, trains baselines, tunes the best model, and returns the AutoMLPipeline instance.
    """
    logs = []
    pipeline = AutoMLPipeline()
    pipeline.target_col = target_col
    
    # 1. Task detection
    task = detect_task_type(df, target_col)
    pipeline.task_type = task
    logs.append(f"AutoML Task Detected: {task.upper()}")
    
    # 2. Separate Features & Target
    if task == "clustering":
        x = df.copy()
    else:
        x = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
    # 3. Clean columns (Feature selection)
    initial_cols = x.columns.tolist()
    cols_to_keep = []
    for col in initial_cols:
        # Check percentage of missing values
        missing_pct = x[col].isnull().sum() / len(x)
        if missing_pct > 0.9:
            logs.append(f"Dropped column '{col}': Too many missing values ({missing_pct:.1%})")
            continue
            
        # Check cardinality
        nunique = x[col].nunique()
        if nunique <= 1:
            logs.append(f"Dropped column '{col}': Single unique value (no variance)")
            continue
            
        if x[col].dtype == object or isinstance(x[col].dtype, pd.CategoricalDtype):
            if nunique / len(x) > 0.95 and nunique > 50:
                logs.append(f"Dropped column '{col}': High cardinality text/categorical (looks like ID)")
                continue
                
        cols_to_keep.append(col)
        
    x = x[cols_to_keep]
    pipeline.feature_cols = cols_to_keep
    
    # Identify numeric and categorical columns
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Gather UI metadata (ranges for slides/selectbox)
    for col in categorical_cols:
        pipeline.categorical_options[col] = sorted([str(v) for v in df[col].dropna().unique()])
    for col in numeric_cols:
        pipeline.numeric_bounds[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median())
        }
        
    logs.append(f"Features for modeling: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical.")
    
    # Fit Preprocessor
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    x_proc = preprocessor.fit_transform(x)
    pipeline.preprocessor = preprocessor
    
    # Extract feature names after preprocessing (useful for feature importances)
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    except Exception:
        cat_feature_names = []
    
    processed_feature_names = numeric_cols + cat_feature_names
    
    # 4. Model Training & Evaluation
    if task == "clustering":
        # Unsupervised Clustering
        logs.append("Running K-Means Clustering AutoML...")
        
        # Test K from 2 to 6, evaluate silhouette
        best_k = 2
        best_score = -1.0
        best_km = None
        
        max_k = min(6, len(x) - 1)
        if max_k >= 2:
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
                labels = km.fit_predict(x_proc)
                score = silhouette_score(x_proc, labels)
                logs.append(f"K-Means with k={k} -> Silhouette Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_km = km
        else:
            best_k = 2
            best_km = KMeans(n_clusters=best_k, random_state=random_state, n_init='auto').fit(x_proc)
            best_score = 0.0
            
        pipeline.best_model = best_km
        pipeline.model_name = f"K-Means Clustering (k={best_k})"
        pipeline.metrics = {"Silhouette Score": best_score, "Clusters": float(best_k)}
        logs.append(f"Optimal cluster count chosen: k={best_k} with Silhouette Score: {best_score:.4f}")
        
    else:
        # Supervised Classification or Regression
        x_train, x_test, y_train, y_test = train_test_split(x_proc, y, test_size=0.2, random_state=random_state)
        
        if task == "classification":
            logs.append("Training Classification Baselines...")
            models = {
                "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=random_state),
                "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
            }
            
            best_score = -1.0
            best_model_name = ""
            best_raw_model = None
            evals = {}
            
            for name, model in models.items():
                try:
                    model.fit(x_train, y_train)
                    preds = model.predict(x_test)
                    
                    # Calculate F1 Macro
                    f1 = f1_score(y_test, preds, average='macro')
                    evals[name] = f1
                    logs.append(f" - {name} -> Test F1-Score (Macro): {f1:.4f}")
                    
                    if f1 > best_score:
                        best_score = f1
                        best_model_name = name
                        best_raw_model = model
                except Exception as e:
                    logs.append(f"Error training {name}: {str(e)}")
                    
            if not best_raw_model:
                raise ValueError("All classification baseline models failed to train.")
                
            logs.append(f"Best Baseline: {best_model_name} (F1: {best_score:.4f})")
            
            # Simple hyperparameter tuning on best model to improve results
            tuned_model = best_raw_model
            if best_model_name == "Random Forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
                logs.append(f"Tuning hyperparameters for {best_model_name}...")
                search = RandomizedSearchCV(tuned_model, param_grid, n_iter=4, cv=3, random_state=random_state, scoring='f1_macro', n_jobs=-1)
                search.fit(x_train, y_train)
                tuned_model = search.best_estimator_
                logs.append(f"Tuned Random Forest Best Params: {search.best_params_}")
            elif best_model_name == "Gradient Boosting":
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5]
                }
                logs.append(f"Tuning hyperparameters for {best_model_name}...")
                search = RandomizedSearchCV(tuned_model, param_grid, n_iter=4, cv=3, random_state=random_state, scoring='f1_macro', n_jobs=-1)
                search.fit(x_train, y_train)
                tuned_model = search.best_estimator_
                logs.append(f"Tuned Gradient Boosting Best Params: {search.best_params_}")
            elif best_model_name == "Logistic Regression":
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['lbfgs', 'saga']
                }
                logs.append(f"Tuning hyperparameters for {best_model_name}...")
                search = RandomizedSearchCV(tuned_model, param_grid, n_iter=4, cv=3, random_state=random_state, scoring='f1_macro', n_jobs=-1)
                search.fit(x_train, y_train)
                tuned_model = search.best_estimator_
                logs.append(f"Tuned Logistic Regression Best Params: {search.best_params_}")
                
            # Re-evaluate best tuned model
            tuned_model.fit(x_train, y_train)
            test_preds = tuned_model.predict(x_test)
            
            # Compute classification evaluation metrics
            accuracy = accuracy_score(y_test, test_preds)
            precision = precision_score(y_test, test_preds, average='macro', zero_division=0)
            recall = recall_score(y_test, test_preds, average='macro', zero_division=0)
            f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
            
            # ROC AUC computation
            roc_auc = 0.5
            if len(np.unique(y_test)) > 1:
                try:
                    if hasattr(tuned_model, "predict_proba"):
                        test_probs = tuned_model.predict_proba(x_test)
                        if len(np.unique(y_test)) == 2:
                            # Binary
                            roc_auc = roc_auc_score(y_test, test_probs[:, 1])
                        else:
                            # Multiclass
                            roc_auc = roc_auc_score(y_test, test_probs, multi_class='ovr', average='macro')
                except Exception as e:
                    logs.append(f"ROC AUC computation failed: {str(e)}")
                    
            pipeline.best_model = tuned_model
            pipeline.model_name = f"Tuned {best_model_name}"
            pipeline.metrics = {
                "Accuracy": float(accuracy),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1-Score": float(f1),
                "ROC-AUC": float(roc_auc)
            }
            logs.append(f"Final Tuned Model Metrics -> Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
            
        else:
            # Regression task
            logs.append("Training Regression Baselines...")
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=random_state),
                "Gradient Boosting": GradientBoostingRegressor(random_state=random_state)
            }
            
            best_score = -np.inf
            best_model_name = ""
            best_raw_model = None
            
            for name, model in models.items():
                try:
                    model.fit(x_train, y_train)
                    preds = model.predict(x_test)
                    
                    r2 = r2_score(y_test, preds)
                    logs.append(f" - {name} -> Test R2-Score: {r2:.4f}")
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model_name = name
                        best_raw_model = model
                except Exception as e:
                    logs.append(f"Error training {name}: {str(e)}")
                    
            if not best_raw_model:
                raise ValueError("All regression baseline models failed to train.")
                
            logs.append(f"Best Baseline: {best_model_name} (R2: {best_score:.4f})")
            
            # Simple tuning
            tuned_model = best_raw_model
            if best_model_name == "Random Forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
                logs.append(f"Tuning hyperparameters for {best_model_name}...")
                search = RandomizedSearchCV(tuned_model, param_grid, n_iter=4, cv=3, random_state=random_state, scoring='r2', n_jobs=-1)
                search.fit(x_train, y_train)
                tuned_model = search.best_estimator_
                logs.append(f"Tuned Random Forest Best Params: {search.best_params_}")
            elif best_model_name == "Gradient Boosting":
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5]
                }
                logs.append(f"Tuning hyperparameters for {best_model_name}...")
                search = RandomizedSearchCV(tuned_model, param_grid, n_iter=4, cv=3, random_state=random_state, scoring='r2', n_jobs=-1)
                search.fit(x_train, y_train)
                tuned_model = search.best_estimator_
                logs.append(f"Tuned Gradient Boosting Best Params: {search.best_params_}")
                
            # Re-evaluate
            tuned_model.fit(x_train, y_train)
            test_preds = tuned_model.predict(x_test)
            
            r2 = r2_score(y_test, test_preds)
            mae = mean_absolute_error(y_test, test_preds)
            rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            
            pipeline.best_model = tuned_model
            pipeline.model_name = f"Tuned {best_model_name}"
            pipeline.metrics = {
                "R-squared": float(r2),
                "MAE": float(mae),
                "RMSE": float(rmse)
            }
            logs.append(f"Final Tuned Model Metrics -> R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
        # 5. Extract Feature Importance for Supervised Models
        try:
            importances = None
            if hasattr(pipeline.best_model, "feature_importances_"):
                importances = pipeline.best_model.feature_importances_
            elif hasattr(pipeline.best_model, "coef_"):
                # For classification, coef_ can be multidimensional (multiclass), so absolute sum/mean coefficients
                coef = pipeline.best_model.coef_
                if coef.ndim > 1:
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)
                    
            if importances is not None and len(importances) == len(processed_feature_names):
                importance_df = pd.DataFrame({
                    "feature": processed_feature_names,
                    "importance": importances
                })
                # Normalize importance to sum up to 1 for readability (already true for Tree models but not linear model coefficients)
                imp_sum = importance_df["importance"].sum()
                if imp_sum > 0:
                    importance_df["importance"] = importance_df["importance"] / imp_sum
                importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
                pipeline.feature_importance_df = importance_df
                logs.append("Extracted feature importances successfully.")
        except Exception as e:
            logs.append(f"Could not extract feature importances: {str(e)}")
            
    return pipeline, logs
