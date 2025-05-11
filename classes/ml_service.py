# ml_service.py
import os
import logging
from typing import Tuple, Dict, List, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from werkzeug.datastructures import FileStorage

# Ensure the 'models' directory exists
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

logger = logging.getLogger(__name__)  # Use module-level logger


class MLService:
    """Encapsulates the machine learning logic for the Flask app."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir

    def _load_dataframe(self, file: FileStorage) -> pd.DataFrame:
        """Loads a DataFrame from a file storage object (CSV or Excel)."""
        filename = file.filename
        if not filename:
            raise ValueError("File has no filename.")
        if filename.endswith(".csv"):
            logger.info(f"Reading CSV file: {filename}")
            df = pd.read_csv(file)
        elif filename.endswith((".xls", ".xlsx")):
            logger.info(f"Reading Excel file: {filename}")
            df = pd.read_excel(file)
        else:
            raise ValueError("Invalid file format. Please upload .csv, .xls, or .xlsx.")
        return df

    def inspect_file(self, file: FileStorage) -> Dict[str, Any]:
        """Reads a file, extracts metadata, and identifies categorical columns."""
        try:
            df = self._load_dataframe(file)
            columns = df.columns.tolist()
            # Ensure serializable types for JSON
            rows = df.head(10).replace({np.nan: None}).values.tolist()
            categorical = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            return {
                "columns": columns,
                "rows": rows,
                "categorical": categorical,
            }
        except Exception as e:
            logger.error(f"Error inspecting file: {e}", exc_info=True)
            raise  # Re-raise the exception to be caught by the route

    def _preprocess_data(
        self,
        df: pd.DataFrame,
        selected_columns: Optional[List[str]],
        missing_values_option: str,
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """Preprocesses the dataframe: selects columns, handles missing values, encodes categoricals."""

        if selected_columns:
            logger.debug(f"Selecting columns: {selected_columns}")
            df = df[selected_columns]
        else:
            logger.debug("Using all columns.")
            selected_columns = df.columns.tolist()  # Use all if none specified

        # Handle missing values
        logger.info(f"Handling missing values using strategy: {missing_values_option}")
        if missing_values_option == "drop":
            df = df.dropna()
        elif missing_values_option == "mean":
            # Fill only numeric columns with mean
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            # Drop rows if non-numeric still have NaN (or handle differently if needed)
            df = df.dropna()
        elif missing_values_option == "median":
            # Fill only numeric columns with median
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            # Drop rows if non-numeric still have NaN
            df = df.dropna()
        else:
            raise ValueError(f"Invalid missing values option: {missing_values_option}")

        if df.empty:
            raise ValueError(
                "DataFrame is empty after handling missing values. Check your data or imputation strategy."
            )

        # Encode categorical columns
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        logger.debug(f"Encoding categorical columns: {categorical_columns}")
        encoders: Dict[str, LabelEncoder] = {}
        for col in categorical_columns:
            # Ensure column is treated as string before encoding
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            # Use fit_transform for training data
            df[col] = le.fit_transform(df[col])
            encoders[col] = le  # Store the fitted encoder

        # Check if target variable (last column) is numeric after encoding
        target_col = df.columns[-1]
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise TypeError(
                f"Target column '{target_col}' is not numeric after preprocessing. Ensure it's suitable for classification or was encoded correctly."
            )

        return df, encoders

    def _save_model(
        self,
        model: RandomForestClassifier,
        encoders: Dict[str, LabelEncoder],
        model_id: str,
    ) -> str:
        """Saves the trained model and encoders."""
        model_path = os.path.join(self.model_dir, f"{model_id}.pkl")
        try:
            joblib.dump((model, encoders), model_path)
            logger.info(f"Model and encoders saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}", exc_info=True)
            raise IOError(f"Could not save model to {model_path}")

    def _load_model(
        self, model_path: str
    ) -> Tuple[RandomForestClassifier, Dict[str, LabelEncoder]]:
        """Loads a model and its associated encoders."""
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        try:
            model, encoders = joblib.load(model_path)
            logger.info(f"Model and encoders loaded from {model_path}")
            if not isinstance(model, RandomForestClassifier):
                raise TypeError("Loaded object is not a RandomForestClassifier model.")
            if not isinstance(encoders, dict):
                raise TypeError("Loaded encoders object is not a dictionary.")
            return model, encoders
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
            raise IOError(f"Could not load model from {model_path}")

    def train(
        self,
        file: FileStorage,
        selected_columns: Optional[List[str]],
        missing_values_option: str,
        num_trees: int,
        max_depth: Optional[int],
    ) -> Dict[str, Any]:
        """Loads data, preprocesses, trains, evaluates, and saves a model."""
        try:
            df = self._load_dataframe(file)
            processed_df, encoders = self._preprocess_data(
                df, selected_columns, missing_values_option
            )

            if processed_df.shape[1] < 2:
                raise ValueError(
                    "Need at least one feature column and one target column after preprocessing."
                )

            X = processed_df.iloc[:, :-1]
            y = processed_df.iloc[:, -1]
            logger.debug(f"Data shape after preprocessing: X={X.shape}, y={y.shape}")

            if X.empty or y.empty:
                raise ValueError(
                    "Features (X) or target (y) is empty after preprocessing and splitting."
                )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=(
                    y
                    if pd.api.types.is_categorical_dtype(y) or y.nunique() > 1
                    else None
                ),
            )
            logger.info(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

            # Train the model
            model = RandomForestClassifier(
                n_estimators=num_trees,
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced",  # Added class_weight
            )
            model.fit(X_train, y_train)
            logger.info("RandomForest model training complete.")

            # Evaluate the model
            accuracy = model.score(X_test, y_test)
            logger.info(f"Model accuracy on test set: {accuracy:.4f}")

            # Save the model and encoders
            model_id = f"random_forest_{num_trees}_{max_depth or 'none'}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
            model_path = self._save_model(model, encoders, model_id)

            return {
                "modelId": model_path,
                "accuracy": accuracy,
                "feature_names": X.columns.tolist(),
            }

        except (ValueError, TypeError, IOError) as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise  # Re-raise specific errors
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during training: {e}", exc_info=True
            )
            raise RuntimeError(
                f"An unexpected error occurred during training: {str(e)}"
            )  # Raise a generic runtime error

    def predict(self, model_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Loads a model and predicts on new input data."""
        try:
            model, encoders = self._load_model(model_path)

            # Convert input dict to DataFrame
            input_df = pd.DataFrame([input_data])
            logger.debug(f"Input data for prediction: {input_df}")

            # Apply learned encoders to input data
            for col, le in encoders.items():
                if col in input_df.columns:
                    # Ensure input is string, handle unseen labels
                    input_value = str(input_df.at[0, col])
                    try:
                        # Use transform for prediction data
                        input_df.at[0, col] = le.transform([input_value])[0]
                    except ValueError:
                        # Handle unseen label: Option 1: Error out
                        raise ValueError(
                            f"Value '{input_value}' in column '{col}' was not seen during training."
                        )
                        # Option 2: Assign a default value (e.g., -1 or np.nan) if your model/pipeline handles it
                        # input_df.at[0, col] = -1 # Or another placeholder
                        # logger.warning(f"Unseen value '{input_value}' in column '{col}'. Assigning placeholder.")
                else:
                    # This case should ideally not happen if input validation is done properly upstream
                    logger.warning(
                        f"Column '{col}' expected by encoder not found in input data."
                    )

            # Ensure columns match model's expected features *after* encoding
            expected_features = model.feature_names_in_
            missing_cols = set(expected_features) - set(input_df.columns)
            extra_cols = set(input_df.columns) - set(expected_features)

            if missing_cols:
                raise ValueError(
                    f"Missing required features in input data: {missing_cols}"
                )
            if extra_cols:
                logger.warning(
                    f"Input data has extra features not used by the model: {extra_cols}"
                )
                input_df = input_df[expected_features]  # Drop extra columns

            # Reorder columns to match training order
            input_df = input_df[expected_features]

            # Handle potential NaNs introduced if imputation is needed for prediction (unlikely here but good practice)
            if input_df.isnull().values.any():
                # Define a strategy or raise error
                raise ValueError(
                    "Input data contains null values after encoding. Prediction cannot proceed."
                )

            # Make predictions
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)

            # Try to inverse transform the prediction if the target was encoded
            # This assumes the target variable's encoder was *not* saved in the `encoders` dict,
            # which is typical as y is usually handled separately. If the target *was* encoded and
            # its encoder saved (e.g., under a special key like '__target__'), you'd load and use it here.
            # For now, returning the encoded prediction.
            final_predictions = predictions.tolist()

            logger.info(
                f"Prediction successful. Predictions: {final_predictions}, Probabilities: {probabilities.tolist()}"
            )
            return {
                "predictions": final_predictions,
                "probabilities": probabilities.tolist(),
            }

        except (FileNotFoundError, IOError, ValueError, TypeError) as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise  # Re-raise specific errors
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during prediction: {e}", exc_info=True
            )
            raise RuntimeError(
                f"An unexpected error occurred during prediction: {str(e)}"
            )
