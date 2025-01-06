import numpy as np
import pandas as pd
from typing import Tuple
from flask import Flask, json, jsonify, render_template, request, redirect, url_for
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from  flask_cors import CORS


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)

template_dir = os.path.abspath("./dist")
app = Flask(__name__, template_folder="dist", static_folder="dist/assets")
CORS(app)

# Serve the homepage
@app.route("/")
def home():
    return render_template("index.html")


# Route for file upload
@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No hay ningun archivo en la solicitud"}), 400

        file = request.files["file"]

        if not file.filename.endswith((".xls", ".xlsx", ".csv")):
            return (
                jsonify(
                    {
                        "error": "Formato de archivo no válido. Cargue un archivo .xls, .xlsx o .csv."
                    }
                ),
                400,
            )

        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        columns = df.columns.tolist()
        rows = df.head(10).values.tolist()

        categorical = [
            col for col in df.select_dtypes(include=["object"]).columns.tolist()
        ]

        response_data = {
            "columns": columns,
            "rows": rows,
            "categorical": categorical,
        }
        response = jsonify(response_data)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def train_model():
    try:
        logging.info("Received a training request")

        # Log request headers and body for debugging
        logging.debug(f"Request headers: {request.headers}")
        logging.debug(f"Request form data: {request.form}")
        logging.debug(f"Request files: {request.files}")

        # Retrieve parameters
        num_trees = int(request.form.get("numTrees", 100))
        max_depth = (
            int(request.form.get("maxDepth", None))
            if request.form.get("maxDepth")
            else None
        )
        # Sanitize the missingValuesOption
        missing_values_option = request.form.get("missingValuesOption", "drop").strip(
            '"'
        )
        selected_columns = request.form.get("selectedColumns", None)
        file = request.files.get("file")

        if not file:
            logging.error("No file provided in the request.")
            return jsonify({"error": "No file provided for training."}), 400

        # Process the uploaded file
        if file.filename.endswith(".csv"):
            logging.info("Processing CSV file.")
            df = pd.read_csv(file)
        else:
            logging.info("Processing Excel file.")
            df = pd.read_excel(file)

        selected_columns = (
            json.loads(selected_columns) if selected_columns else df.columns.tolist()
        )
        logging.debug(f"Selected columns for training: {selected_columns}")
        df = df[selected_columns]

        # Handle missing values
        if missing_values_option == "drop":
            logging.info("Dropping rows with missing values.")
            df = df.dropna()
        elif missing_values_option == "mean":
            logging.info("Filling missing values with column means.")
            df = df.fillna(df.mean(numeric_only=True))
        elif missing_values_option == "median":
            logging.info("Filling missing values with column medians.")
            df = df.fillna(df.median(numeric_only=True))
        else:
            logging.error(f"Invalid missing values option: {missing_values_option}")
            return jsonify({"error": "Invalid missing values option."}), 400

        # Encode categorical columns
        categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
        logging.debug(f"Categorical columns: {categorical_columns}")
        encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le  # Save the encoder

        # Split the data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        logging.debug("Data split into features and target.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logging.info("Training and test data split.")

        # Train the model
        model = RandomForestClassifier(
            n_estimators=num_trees, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        logging.info("RandomForest model trained successfully.")

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        logging.info(f"Model accuracy: {accuracy}")

        # Save the model
        model_path = f"models/random_forest_{num_trees}_{max_depth or 'none'}.pkl"
        joblib.dump((model, encoders), model_path)
        logging.info(f"Model saved to {model_path}")

        response = jsonify({"modelId": model_path, "accuracy": accuracy})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 200

    except Exception as e:
        logging.exception("An error occurred during the training process.")
        return jsonify({"error": str(e)}), 500


@app.route("/api/test", methods=["POST"])
def test_model():
    try:
        logging.info("Received a prediction request.")
        model_path = request.form.get("modelPath")
        input_data = request.form.get("inputData")

        if not model_path or not input_data:
            return jsonify({"error": "Missing required fields."}), 400

        input_data = json.loads(input_data) if isinstance(input_data, str) else input_data
        input_df = pd.DataFrame([input_data])

        # Load model and encoders
        try:
            model, encoders = joblib.load(model_path)
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return jsonify({"error": "Failed to load model."}), 500

        # Encode categorical features
        for col, le in encoders.items():
            if col in input_df:
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError:
                    return jsonify({"error": f"Invalid value in column '{col}'."}), 400

        # Validate features
        expected_features = model.feature_names_in_
        if not all(feature in input_df.columns for feature in expected_features):
            return jsonify({"error": "Input features do not match model requirements."}), 400

        input_df = input_df[expected_features]

        # Make predictions
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df).tolist()

        return jsonify({"predictions": predictions.tolist(), "probabilities": probabilities}), 200

    except Exception as e:
        logging.exception("An error occurred during prediction.")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
