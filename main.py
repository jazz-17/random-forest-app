import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import logging
import os
import json
import pandas as pd

# Asumo que estas clases están en una carpeta 'classes' relativa a este script
from classes.file_wrapper import FileObjectWrapper
from classes.ml_service import MLService, MODEL_DIR


# Configuración básica de logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MLAppGUI(tk.Tk):
    def __init__(self, ml_service):
        super().__init__()
        self.ml_service = ml_service

        self.title("Interfaz Gráfica para Servicio de ML")  # Traducido
        self.geometry("900x700")

        # --- Inicializar StringVars de Tkinter ---
        self.train_file_path = tk.StringVar()
        self.model_file_path_predict = tk.StringVar()

        self.num_trees_var = tk.StringVar(value="100")
        self.max_depth_var = tk.StringVar(value="")
        self.missing_values_var = tk.StringVar(
            value="drop"
        )  # Opciones técnicas se mantienen

        # --- PanedWindow Principal para secciones redimensionables ---
        main_paned_window = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- PanedWindow Superior para Secciones de Entrada ---
        top_paned_window = ttk.PanedWindow(main_paned_window, orient=tk.HORIZONTAL)
        main_paned_window.add(top_paned_window, weight=3)

        # --- Frame Inferior para Logs/Resultados ---
        results_frame = ttk.LabelFrame(
            main_paned_window,
            text="Registro de Acciones / Resultados Detallados",  # Traducido
        )
        main_paned_window.add(results_frame, weight=1)

        self.results_text = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=8
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.configure(state="disabled")  # Hacerlo de solo lectura

        # --- Crear secciones funcionales ---
        train_frame = self._create_train_frame(top_paned_window)
        predict_frame = self._create_predict_frame(top_paned_window)

        top_paned_window.add(train_frame, weight=1)
        top_paned_window.add(predict_frame, weight=1)

    def _create_train_frame(self, parent):
        frame = ttk.LabelFrame(
            parent,
            text="Entrenar Modelo e Inspeccionar Datos",
            padding=(10, 5),  # Traducido
        )
        row_idx = 0

        # --- Selección de Archivo ---
        ttk.Button(
            frame,
            text="Seleccionar Datos de Entrenamiento (.xls, .xlsx, .csv)",  # Traducido
            command=self._select_train_file_and_inspect,
        ).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="ew")
        row_idx += 1
        ttk.Label(frame, textvariable=self.train_file_path, wraplength=280).grid(
            row=row_idx, column=0, columnspan=2, pady=2, sticky="w"
        )
        row_idx += 1

        # --- Área de Resultados de Inspección ---
        ttk.Label(frame, text="Resultados de Inspección de Datos:").grid(  # Traducido
            row=row_idx, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )
        row_idx += 1
        self.train_file_inspection_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, height=7, width=40
        )
        self.train_file_inspection_text.grid(
            row=row_idx, column=0, columnspan=2, pady=2, sticky="nsew"
        )
        row_idx += 1
        self.train_file_inspection_text.insert(
            tk.END,
            "Seleccione un archivo de entrenamiento para ver sus detalles aquí.",  # Traducido
        )
        self.train_file_inspection_text.configure(state="disabled")

        # --- Parámetros de Entrenamiento ---
        ttk.Label(frame, text="Núm. Árboles:").grid(  # Traducido
            row=row_idx, column=0, sticky="w", pady=2
        )
        ttk.Entry(frame, textvariable=self.num_trees_var).grid(
            row=row_idx, column=1, sticky="ew", pady=2
        )
        row_idx += 1

        ttk.Label(
            frame, text="Profundidad Máx. (vacío para Ninguna):"
        ).grid(  # Traducido
            row=row_idx, column=0, sticky="w", pady=2
        )
        ttk.Entry(frame, textvariable=self.max_depth_var).grid(
            row=row_idx, column=1, sticky="ew", pady=2
        )
        row_idx += 1

        ttk.Label(frame, text="Opción Valores Faltantes:").grid(  # Traducido
            row=row_idx, column=0, sticky="w", pady=2
        )
        missing_options = ["drop", "mean", "median"]  # Opciones técnicas
        # Traducción de opciones para el ComboBox (si se desea)
        # missing_options_display = {"drop": "Eliminar", "mean": "Media", "median": "Mediana"}
        # self.missing_values_var_display = tk.StringVar(value=missing_options_display["drop"])
        ttk.Combobox(
            frame,
            textvariable=self.missing_values_var,  # Usar la variable original para la lógica
            values=missing_options,  # Usar las opciones originales para la lógica
            # values=list(missing_options_display.values()), # Para mostrar en español
            state="readonly",
        ).grid(row=row_idx, column=1, sticky="ew", pady=2)
        row_idx += 1

        ttk.Button(
            frame, text="Entrenar Modelo", command=self._train_model
        ).grid(  # Traducido
            row=row_idx, column=0, columnspan=2, pady=10, sticky="ew"
        )

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(3, weight=1)
        return frame

    def _create_predict_frame(self, parent):
        frame = ttk.LabelFrame(
            parent, text="Predecir con Modelo", padding=(10, 5)
        )  # Traducido
        row_idx = 0

        ttk.Button(
            frame,
            text="Seleccionar Archivo de Modelo",
            command=self._select_model_file_predict,  # Traducido
        ).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="ew")
        row_idx += 1
        ttk.Label(
            frame, textvariable=self.model_file_path_predict, wraplength=280
        ).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w")
        row_idx += 1

        ttk.Label(frame, text="Datos de Entrada (objeto JSON):").grid(  # Traducido
            row=row_idx, column=0, sticky="nw", pady=2
        )
        row_idx += 1
        self.input_data_text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            height=10,  # << INCREASED FROM 5 TO 10 (or any value you prefer)
            width=30,  # You can also adjust width if needed
        )
        self.input_data_text.grid(
            row=row_idx, column=0, columnspan=2, sticky="ew", pady=2
        )
        row_idx += 1
        # You might want to update the default JSON to reflect the new height
        # or provide a more representative example if it's very long.
        # For example, using the heart.csv structure:
        default_json_input = """{
  "Age": 40,
  "Sex": "M",
  "ChestPainType": "ATA",
  "RestingBP": 140,
  "Cholesterol": 289,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 172,
  "ExerciseAngina": "N",
  "Oldpeak": 0.0,
  "ST_Slope": "Up"
}"""
        self.input_data_text.insert(tk.END, default_json_input)

        ttk.Button(
            frame, text="Predecir", command=self._predict_model
        ).grid(  # Traducido
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
        # Los títulos de los diálogos del sistema operativo pueden estar ya en el idioma del SO.
        # Pero podemos proveer títulos en español aquí.
        filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if filepath:
            variable_to_set.set(filepath)
            self._log_to_gui(f"Archivo seleccionado: {filepath}")  # Traducido
            return filepath
        return None

    def _select_train_file_and_inspect(self):
        filepath = self._select_file_dialog(
            self.train_file_path,
            "Seleccionar Archivo de Datos de Entrenamiento",  # Traducido
            (
                ("Archivos Excel", "*.xls *.xlsx"),  # Traducido
                ("Archivos CSV", "*.csv"),  # Traducido
                ("Todos los archivos", "*.*"),  # Traducido
            ),
        )
        if filepath:
            if not filepath.lower().endswith((".xls", ".xlsx", ".csv")):
                messagebox.showerror(
                    "Error",
                    "Formato de archivo no válido. Por favor, seleccione .xls, .xlsx, o .csv.",  # Traducido
                )
                self._update_inspection_text(
                    "Formato de archivo seleccionado no válido.",
                    is_error=True,  # Traducido
                )
                self.train_file_path.set("")
                return

            try:
                self._log_to_gui(
                    f"Inspeccionando archivo de datos de entrenamiento: {filepath}..."
                )  # Traducido
                self._update_inspection_text(
                    f"Inspeccionando {os.path.basename(filepath)}..."  # Traducido
                )
                with FileObjectWrapper(filepath) as f_wrapper:
                    result = self.ml_service.inspect_file(f_wrapper)

                inspection_summary = (
                    f"Archivo: {f_wrapper.filename}\n"  # Traducido (parcial)
                )
                inspection_summary += "--------------------\n"
                inspection_summary += json.dumps(result, indent=2)

                self._update_inspection_text(inspection_summary)
                self._log_to_gui(
                    f"Inspección exitosa para {f_wrapper.filename}."
                )  # Traducido

            except ValueError as e:
                messagebox.showerror("Error de Inspección", str(e))  # Traducido
                self._log_to_gui(
                    f"Error de valor durante la inspección: {e}", "ERROR"
                )  # Traducido
                self._update_inspection_text(
                    f"Error inspeccionando archivo: {e}", is_error=True  # Traducido
                )
            except Exception as e:
                messagebox.showerror(
                    "Error de Inspección",
                    f"Ocurrió un error inesperado: {str(e)}",  # Traducido
                )
                self._log_to_gui(
                    f"Error inesperado durante la inspección de {filepath}: {e}",
                    "ERROR",  # Traducido
                )
                self._update_inspection_text(
                    f"Error inesperado inspeccionando archivo: {e}",
                    is_error=True,  # Traducido
                )
                logger.exception(
                    f"Error inesperado durante la inspección del archivo para {filepath}"
                )

    def _select_model_file_predict(self):
        self._select_file_dialog(
            self.model_file_path_predict,
            "Seleccionar Archivo de Modelo",  # Traducido
            (
                ("Modelo Pickled", "*.pkl"),  # Traducido
                ("Modelo Joblib", "*.joblib"),  # Traducido
                ("Todos los archivos", "*.*"),  # Traducido
            ),
        )

    def _train_model(self):
        filepath = self.train_file_path.get()
        if not filepath:
            messagebox.showerror(
                "Error", "No se ha seleccionado ningún archivo para entrenar."
            )  # Traducido
            return

        try:
            num_trees_str = self.num_trees_var.get()
            max_depth_str = self.max_depth_var.get()
            missing_values_option = self.missing_values_var.get()

            num_trees = int(num_trees_str)
            if num_trees <= 0:
                raise ValueError("El número de árboles debe ser positivo.")  # Traducido

            max_depth = None
            if (
                max_depth_str
                and max_depth_str.lower()
                != "none"  # Mantener "none" como palabra clave si el servicio lo espera
                and max_depth_str.strip() != ""
            ):
                max_depth = int(max_depth_str)
                if max_depth <= 0:
                    raise ValueError(
                        "La profundidad máxima debe ser positiva."
                    )  # Traducido

            selected_columns = None
            self._log_to_gui(
                f"Iniciando entrenamiento con archivo: {filepath}..."
            )  # Traducido
            self._log_to_gui(
                f"Parámetros: núm_árboles={num_trees}, prof_máx={max_depth}, "  # Traducido
                f"opc_faltantes={missing_values_option}, columnas={selected_columns}"
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
                f"Entrenamiento exitoso. ID del Modelo: {result.get('modelId', 'N/D')}, "  # Traducido
                f"Precisión (Accuracy): {result.get('accuracy', float('nan')):.4f}"  # Traducido
            )
            self._log_to_gui(
                "Resultado completo del entrenamiento:\n" + json.dumps(result, indent=2)
            )  # Traducido

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            messagebox.showerror(
                "Error en Parámetros de Entrenamiento", str(e)
            )  # Traducido
            self._log_to_gui(
                f"Error de parámetro durante el entrenamiento: {e}", "ERROR"
            )  # Traducido
        except (FileNotFoundError, IOError) as e:
            messagebox.showerror("Error de Archivo", str(e))  # Traducido
            self._log_to_gui(
                f"Error de archivo durante el entrenamiento: {e}", "ERROR"
            )  # Traducido
        except RuntimeError as e:
            messagebox.showerror("Error de Ejecución", str(e))  # Traducido
            self._log_to_gui(
                f"Error de ejecución durante el entrenamiento: {e}", "ERROR"
            )  # Traducido
            logger.exception("Error de ejecución durante el entrenamiento del modelo")
        except Exception as e:
            messagebox.showerror(
                "Error de Entrenamiento",
                f"Ocurrió un error inesperado: {str(e)}",  # Traducido
            )
            self._log_to_gui(
                f"Error inesperado durante el entrenamiento: {e}", "ERROR"
            )  # Traducido
            logger.exception("Error inesperado durante el entrenamiento del modelo")

    def _predict_model(self):
        model_path = self.model_file_path_predict.get()
        input_data_json = self.input_data_text.get("1.0", tk.END).strip()

        if not model_path or not input_data_json:
            messagebox.showerror(
                "Error", "Falta la ruta del modelo o los datos de entrada."
            )  # Traducido
            return

        try:
            input_data = json.loads(input_data_json)
            if not isinstance(input_data, dict):
                raise ValueError(
                    "Los datos de entrada deben ser un objeto JSON (diccionario)."
                )  # Traducido

            self._log_to_gui(f"Prediciendo con modelo: {model_path}...")  # Traducido
            self._log_to_gui(f"Datos de entrada: {json.dumps(input_data)}")  # Traducido

            result = self.ml_service.predict(
                model_path=model_path, input_data=input_data
            )

            self._log_to_gui(
                f"Predicción exitosa para el modelo {model_path}."
            )  # Traducido
            self._log_to_gui(
                "Resultado de la predicción:\n" + json.dumps(result, indent=2)
            )  # Traducido

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            messagebox.showerror(
                "Error en Datos de Entrada de Predicción", str(e)
            )  # Traducido
            self._log_to_gui(
                f"Error de entrada durante la predicción: {e}", "ERROR"
            )  # Traducido
        except (FileNotFoundError, IOError) as e:
            messagebox.showerror("Error de Archivo", str(e))  # Traducido
            self._log_to_gui(
                f"Error de archivo durante la predicción: {e}", "ERROR"
            )  # Traducido
        except RuntimeError as e:
            messagebox.showerror("Error de Ejecución", str(e))  # Traducido
            self._log_to_gui(
                f"Error de ejecución durante la predicción: {e}", "ERROR"
            )  # Traducido
            logger.exception("Error de ejecución durante la predicción")
        except Exception as e:
            messagebox.showerror(
                "Error de Predicción",
                f"Ocurrió un error inesperado: {str(e)}",  # Traducido
            )
            self._log_to_gui(
                f"Error inesperado durante la predicción: {e}", "ERROR"
            )  # Traducido
            logger.exception("Error inesperado durante la predicción")


if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR)
            logger.info(f"Directorio de modelos creado: {MODEL_DIR}")  # Traducido
        except OSError as e:
            logger.error(
                f"No se pudo crear el directorio de modelos {MODEL_DIR}: {e}"
            )  # Traducido

    ml_service_instance = MLService(model_dir=MODEL_DIR)
    app = MLAppGUI(ml_service=ml_service_instance)
    app.mainloop()
