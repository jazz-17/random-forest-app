# app.py (or your main flask file)
import logging
import os
from flask import Flask, json, jsonify, render_template, request, redirect, url_for
from flask_cors import CORS
import pandas as pd  # Keep for potential type checking if needed

# Import the service
from ml_service import MLService, MODEL_DIR  # Import MODEL_DIR if needed elsewhere

# Configure logging (basic setup)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Get logger for this module
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="dist", static_folder="dist/assets")
CORS(app)  # Enable CORS for all routes

# Instantiate the ML Service
# Can be global if stateless, or created per request if stateful
ml_service = MLService(model_dir=MODEL_DIR)


# Serve the homepage
@app.route("/")
def home():
    # No change needed here
    return render_template("index.html")


# Route for file upload inspection
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        logger.warning("Upload request received with no file part.")
        return jsonify({"error": "No hay ningun archivo en la solicitud"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        logger.warning("Upload request received with empty filename.")
        return jsonify({"error": "No file selected."}), 400

    # Basic filename check (service layer does more robust format check)
    if not file.filename.lower().endswith((".xls", ".xlsx", ".csv")):
        logger.warning(f"Invalid file format uploaded: {file.filename}")
        return (
            jsonify(
                {
                    "error": "Formato de archivo no válido. Cargue un archivo .xls, .xlsx o .csv."
                }
            ),
            400,
        )

    try:
        # Delegate the core logic to the service
        result = ml_service.inspect_file(file)
        logger.info(f"File inspection successful for {file.filename}")
        return jsonify(result), 200

    except ValueError as e:  # Catch specific errors from service
        logger.error(f"Value error during file inspection: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(
            f"Unexpected error during file inspection for {file.filename}"
        )  # Log full traceback
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/api/train", methods=["POST"])
def train_model_route():
    logger.info("Received request for model training.")
    try:
        # --- Parameter Extraction and Validation ---
        if "file" not in request.files:
            logger.error("Training request missing file.")
            return jsonify({"error": "No file provided for training."}), 400
        file = request.files["file"]
        if not file or file.filename == "":
            logger.error("Training request received with empty filename.")
            return jsonify({"error": "No file selected for training."}), 400

        num_trees_str = request.form.get("numTrees", "100")
        max_depth_str = request.form.get("maxDepth")
        missing_values_option = request.form.get("missingValuesOption", "drop").strip(
            '"'
        )
        selected_columns_json = request.form.get("selectedColumns")

        try:
            num_trees = int(num_trees_str)
            if num_trees <= 0:
                raise ValueError("Number of trees must be positive.")
        except ValueError:
            logger.error(f"Invalid numTrees value: {num_trees_str}")
            return jsonify({"error": "Invalid value for number of trees."}), 400

        max_depth = None
        if max_depth_str and max_depth_str.lower() != "none" and max_depth_str != "":
            try:
                max_depth = int(max_depth_str)
                if max_depth <= 0:
                    raise ValueError("Max depth must be positive.")
            except ValueError:
                logger.error(f"Invalid maxDepth value: {max_depth_str}")
                return jsonify({"error": "Invalid value for max depth."}), 400

        selected_columns = None
        if selected_columns_json and selected_columns_json.lower() != "null":
            try:
                selected_columns = json.loads(selected_columns_json)
                if not isinstance(selected_columns, list):
                    raise ValueError("selectedColumns must be a list.")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    f"Invalid selectedColumns JSON: {selected_columns_json} - {e}"
                )
                return jsonify({"error": "Invalid format for selected columns."}), 400

        # --- Delegate to Service ---
        logger.debug(
            f"Training parameters: num_trees={num_trees}, max_depth={max_depth}, missing_opt={missing_values_option}, columns={selected_columns}"
        )
        result = ml_service.train(
            file=file,
            selected_columns=selected_columns,
            missing_values_option=missing_values_option,
            num_trees=num_trees,
            max_depth=max_depth,
        )
        logger.info(
            f"Training successful. Model ID: {result['modelId']}, Accuracy: {result['accuracy']:.4f}"
        )
        return jsonify(result), 200

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        IOError,
    ) as e:  # Catch specific errors
        logger.error(f"Error during model training request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400  # Or 500 depending on error type
    except RuntimeError as e:  # Catch unexpected runtime errors from service
        logger.exception("Runtime error during model training request.")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception("Unexpected error during model training request.")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500


@app.route("/api/test", methods=["POST"])
def test_model_route():
    logger.info("Received request for model prediction.")
    try:
        # --- Parameter Extraction and Validation ---
        model_path = request.form.get("modelPath")
        input_data_json = request.form.get("inputData")

        if not model_path or not input_data_json:
            logger.warning("Prediction request missing modelPath or inputData.")
            return (
                jsonify({"error": "Missing required fields: modelPath and inputData."}),
                400,
            )

        try:
            input_data = json.loads(input_data_json)
            if not isinstance(input_data, dict):
                raise ValueError("inputData must be a JSON object (dictionary).")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid inputData JSON: {input_data_json} - {e}")
            return (
                jsonify(
                    {"error": "Invalid format for input data. Expected a JSON object."}
                ),
                400,
            )

        # --- Delegate to Service ---
        logger.debug(
            f"Prediction request: model_path={model_path}, input_data={input_data}"
        )
        result = ml_service.predict(model_path=model_path, input_data=input_data)
        logger.info(f"Prediction successful for model {model_path}.")
        return jsonify(result), 200

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        IOError,
    ) as e:  # Catch specific errors
        logger.error(f"Error during prediction request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400  # Or 500
    except RuntimeError as e:  # Catch unexpected runtime errors from service
        logger.exception("Runtime error during prediction request.")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception("Unexpected error during prediction request.")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # Make sure the model directory exists (redundant if MLService is imported, but safe)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    # Use debug=False for production, True for development
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "False").lower() == "true",
        host="0.0.0.0",
        port=5000,
    )
