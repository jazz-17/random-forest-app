import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import logging
import os
import json
import pandas as pd  # Keep for type hints or if MLService returns DataFrames directly

# Import the service
from ml_service import MLService, MODEL_DIR

# Configure logging (basic setup)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Helper File Wrapper ---
class FileObjectWrapper:
    def __init__(self, file_path):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self._file = None

    def open(self, mode="rb"):
        self._file = open(self.file_path, mode)
        return self

    def read(self, *args, **kwargs):
        return self._file.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._file.seek(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self._file.tell(*args, **kwargs)

    def close(self):
        if self._file:
            self._file.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MLAppGUI(tk.Tk):
    def __init__(self, ml_service):
        super().__init__()
        self.ml_service = ml_service

        self.title("ML Service GUI")
        self.geometry("900x700")  # Adjusted height slightly

        # --- Initialize Tkinter StringVars ---
        self.train_file_path = tk.StringVar()
        self.model_file_path_predict = tk.StringVar()

        self.num_trees_var = tk.StringVar(value="100")
        self.max_depth_var = tk.StringVar(value="")
        self.missing_values_var = tk.StringVar(value="drop")
        self.selected_columns_var = tk.StringVar(value="")

        # --- Main PanedWindow for resizable sections ---
        main_paned_window = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Top PanedWindow for Input Sections ---
        top_paned_window = ttk.PanedWindow(main_paned_window, orient=tk.HORIZONTAL)
        main_paned_window.add(top_paned_window, weight=3)

        # --- Bottom Frame for Logs/Results ---
        results_frame = ttk.LabelFrame(
            main_paned_window, text="Action Log / Detailed Results"
        )
        main_paned_window.add(results_frame, weight=1)

        self.results_text = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=8
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Create functional sections ---
        # REMOVED: inspect_frame = self._create_inspect_frame(top_paned_window)
        train_frame = self._create_train_frame(top_paned_window)
        predict_frame = self._create_predict_frame(top_paned_window)

        # REMOVED: top_paned_window.add(inspect_frame, weight=1)
        top_paned_window.add(
            train_frame, weight=1
        )  # Train frame might need more relative weight now
        top_paned_window.add(predict_frame, weight=1)

    def _create_train_frame(self, parent):
        frame = ttk.LabelFrame(
            parent, text="Train Model & Inspect Data", padding=(10, 5)
        )
        row_idx = 0

        # --- File Selection ---
        ttk.Button(
            frame,
            text="Select Training Data (.xls, .xlsx, .csv)",
            command=self._select_train_file_and_inspect,
        ).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="ew")
        row_idx += 1
        ttk.Label(frame, textvariable=self.train_file_path, wraplength=280).grid(
            row=row_idx, column=0, columnspan=2, pady=2, sticky="w"
        )
        row_idx += 1

        # --- Inspection Results Area ---
        ttk.Label(frame, text="Data Inspection Results:").grid(
            row=row_idx, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        row_idx += 1
        self.train_file_inspection_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, height=7, width=40
        )  # Adjusted height
        self.train_file_inspection_text.grid(
            row=row_idx, column=0, columnspan=2, pady=2, sticky="nsew"
        )
        row_idx += 1
        self.train_file_inspection_text.insert(
            tk.END, "Select a training file to see its details here."
        )
        self.train_file_inspection_text.configure(
            state="disabled"
        )  # Make read-only initially

        # --- Training Parameters ---
        ttk.Label(frame, text="Num Trees:").grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        ttk.Entry(frame, textvariable=self.num_trees_var).grid(
            row=row_idx, column=1, sticky="ew", pady=2
        )
        row_idx += 1

        ttk.Label(frame, text="Max Depth (blank for None):").grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        ttk.Entry(frame, textvariable=self.max_depth_var).grid(
            row=row_idx, column=1, sticky="ew", pady=2
        )
        row_idx += 1

        ttk.Label(frame, text="Missing Values Option:").grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        missing_options = ["drop", "mean", "median"]
        ttk.Combobox(
            frame,
            textvariable=self.missing_values_var,
            values=missing_options,
            state="readonly",
        ).grid(row=row_idx, column=1, sticky="ew", pady=2)
        row_idx += 1

        ttk.Label(frame, text="Selected Columns (JSON list or blank):").grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        ttk.Entry(frame, textvariable=self.selected_columns_var).grid(
            row=row_idx, column=1, sticky="ew", pady=2
        )
        row_idx += 1

        ttk.Button(frame, text="Train Model", command=self._train_model).grid(
            row=row_idx, column=0, columnspan=2, pady=10, sticky="ew"
        )

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(
            3, weight=1
        )  # Allow inspection text area to expand vertically
        return frame

    def _create_predict_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="Predict with Model", padding=(10, 5))
        row_idx = 0

        ttk.Button(
            frame, text="Select Model File", command=self._select_model_file_predict
        ).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="ew")
        row_idx += 1
        ttk.Label(
            frame, textvariable=self.model_file_path_predict, wraplength=280
        ).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w")
        row_idx += 1

        ttk.Label(frame, text="Input Data (JSON object):").grid(
            row=row_idx, column=0, sticky="nw", pady=2
        )
        row_idx += 1
        self.input_data_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, height=5, width=30
        )
        self.input_data_text.grid(
            row=row_idx, column=0, columnspan=2, sticky="ew", pady=2
        )
        row_idx += 1
        self.input_data_text.insert(
            tk.END, '{\n  "feature1": "value1",\n  "feature2": 0\n}'
        )

        ttk.Button(frame, text="Predict", command=self._predict_model).grid(
            row=row_idx, column=0, columnspan=2, pady=10, sticky="ew"
        )

        frame.columnconfigure(0, weight=1)
        return frame

    def _log_to_gui(self, message, level="INFO"):
        self.results_text.configure(state="normal")
        self.results_text.insert(tk.END, f"[{level}] {message}\n")
        self.results_text.see(tk.END)
        self.results_text.configure(state="disabled")
        if level.upper() in ["ERROR", "WARNING", "CRITICAL"]:
            logger.log(logging.getLevelName(level.upper()), message)
        else:
            logger.info(message)

    def _update_inspection_text(self, content, is_error=False):
        self.train_file_inspection_text.configure(state="normal")
        self.train_file_inspection_text.delete("1.0", tk.END)
        self.train_file_inspection_text.insert(tk.END, content)
        self.train_file_inspection_text.configure(state="disabled")

    def _select_file_dialog(self, variable_to_set, title, filetypes):
        filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if filepath:
            variable_to_set.set(filepath)
            self._log_to_gui(f"Selected file: {filepath}")
            return filepath
        return None

    def _select_train_file_and_inspect(self):
        filepath = self._select_file_dialog(
            self.train_file_path,
            "Select Training Data File",
            (
                ("Excel files", "*.xls *.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ),
        )
        if filepath:
            if not filepath.lower().endswith((".xls", ".xlsx", ".csv")):
                messagebox.showerror(
                    "Error", "Invalid file format. Please select .xls, .xlsx, or .csv."
                )
                self._update_inspection_text(
                    "Invalid file format selected.", is_error=True
                )
                self.train_file_path.set("")  # Clear invalid path
                return

            try:
                self._log_to_gui(f"Inspecting training data file: {filepath}...")
                self._update_inspection_text(
                    f"Inspecting {os.path.basename(filepath)}..."
                )
                with FileObjectWrapper(filepath) as f_wrapper:
                    result = self.ml_service.inspect_file(f_wrapper)

                inspection_summary = f"File: {f_wrapper.filename}\n"
                inspection_summary += "--------------------\n"
                inspection_summary += json.dumps(result, indent=2)

                self._update_inspection_text(inspection_summary)
                self._log_to_gui(f"Inspection successful for {f_wrapper.filename}.")

            except ValueError as e:
                messagebox.showerror("Inspection Error", str(e))
                self._log_to_gui(f"Value error during inspection: {e}", "ERROR")
                self._update_inspection_text(
                    f"Error inspecting file: {e}", is_error=True
                )
            except Exception as e:
                messagebox.showerror(
                    "Inspection Error", f"An unexpected error occurred: {str(e)}"
                )
                self._log_to_gui(
                    f"Unexpected error during inspection for {filepath}: {e}", "ERROR"
                )
                self._update_inspection_text(
                    f"Unexpected error inspecting file: {e}", is_error=True
                )
                logger.exception(
                    f"Unexpected error during file inspection for {filepath}"
                )

    def _select_model_file_predict(self):
        self._select_file_dialog(
            self.model_file_path_predict,
            "Select Model File",
            (
                ("Pickled Model", "*.pkl"),
                ("Joblib Model", "*.joblib"),
                ("All files", "*.*"),
            ),
        )

    def _train_model(self):
        filepath = self.train_file_path.get()
        if not filepath:
            messagebox.showerror("Error", "No file selected for training.")
            return

        try:
            num_trees_str = self.num_trees_var.get()
            max_depth_str = self.max_depth_var.get()
            missing_values_option = self.missing_values_var.get()
            selected_columns_json = self.selected_columns_var.get()

            num_trees = int(num_trees_str)
            if num_trees <= 0:
                raise ValueError("Number of trees must be positive.")

            max_depth = None
            if (
                max_depth_str
                and max_depth_str.lower() != "none"
                and max_depth_str.strip() != ""
            ):
                max_depth = int(max_depth_str)
                if max_depth <= 0:
                    raise ValueError("Max depth must be positive.")

            selected_columns = None
            if (
                selected_columns_json
                and selected_columns_json.lower() != "null"
                and selected_columns_json.strip() != ""
            ):
                selected_columns = json.loads(selected_columns_json)
                if not isinstance(selected_columns, list):
                    raise ValueError("Selected columns must be a JSON list.")

            self._log_to_gui(f"Starting training with file: {filepath}...")
            self._log_to_gui(
                f"Params: num_trees={num_trees}, max_depth={max_depth}, "
                f"missing_opt={missing_values_option}, columns={selected_columns}"
            )

            with FileObjectWrapper(filepath) as f_wrapper:
                result = self.ml_service.train(
                    file=f_wrapper,
                    selected_columns=selected_columns,
                    missing_values_option=missing_values_option,
                    num_trees=num_trees,
                    max_depth=max_depth,
                )
            self._log_to_gui(
                f"Training successful. Model ID: {result.get('modelId', 'N/A')}, "
                f"Accuracy: {result.get('accuracy', float('nan')):.4f}"
            )
            self._log_to_gui("Full training result:\n" + json.dumps(result, indent=2))

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            messagebox.showerror("Training Parameter Error", str(e))
            self._log_to_gui(f"Parameter error during training: {e}", "ERROR")
        except (FileNotFoundError, IOError) as e:
            messagebox.showerror("File Error", str(e))
            self._log_to_gui(f"File error during training: {e}", "ERROR")
        except RuntimeError as e:
            messagebox.showerror("Runtime Error", str(e))
            self._log_to_gui(f"Runtime error during training: {e}", "ERROR")
            logger.exception("Runtime error during model training")
        except Exception as e:
            messagebox.showerror(
                "Training Error", f"An unexpected error occurred: {str(e)}"
            )
            self._log_to_gui(f"Unexpected error during training: {e}", "ERROR")
            logger.exception("Unexpected error during model training")

    def _predict_model(self):
        model_path = self.model_file_path_predict.get()
        input_data_json = self.input_data_text.get("1.0", tk.END).strip()

        if not model_path or not input_data_json:
            messagebox.showerror("Error", "Missing model path or input data.")
            return

        try:
            input_data = json.loads(input_data_json)
            if not isinstance(input_data, dict):
                raise ValueError("Input data must be a JSON object (dictionary).")

            self._log_to_gui(f"Predicting with model: {model_path}...")
            self._log_to_gui(f"Input data: {json.dumps(input_data)}")

            result = self.ml_service.predict(
                model_path=model_path, input_data=input_data
            )

            self._log_to_gui(f"Prediction successful for model {model_path}.")
            self._log_to_gui("Prediction result:\n" + json.dumps(result, indent=2))

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            messagebox.showerror("Prediction Input Error", str(e))
            self._log_to_gui(f"Input error during prediction: {e}", "ERROR")
        except (FileNotFoundError, IOError) as e:
            messagebox.showerror("File Error", str(e))
            self._log_to_gui(f"File error during prediction: {e}", "ERROR")
        except RuntimeError as e:
            messagebox.showerror("Runtime Error", str(e))
            self._log_to_gui(f"Runtime error during prediction: {e}", "ERROR")
            logger.exception("Runtime error during prediction")
        except Exception as e:
            messagebox.showerror(
                "Prediction Error", f"An unexpected error occurred: {str(e)}"
            )
            self._log_to_gui(f"Unexpected error during prediction: {e}", "ERROR")
            logger.exception("Unexpected error during prediction")


if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR)
            logger.info(f"Created model directory: {MODEL_DIR}")
        except OSError as e:
            logger.error(f"Could not create model directory {MODEL_DIR}: {e}")

    ml_service_instance = MLService(model_dir=MODEL_DIR)
    app = MLAppGUI(ml_service=ml_service_instance)
    app.mainloop()
