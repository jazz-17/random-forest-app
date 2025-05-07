<script setup>
import { ref, computed } from "vue";
import { RouterLink, useRouter } from "vue-router";
import { modelStore } from "../store/modelStore";
const router = useRouter();
const store = modelStore();

const fileInput = ref(null);
const uploadedFileName = ref(null);
const datasetPreview = ref(null);
const missingValuesOption = ref("drop");
const selectedColumns = ref([]);
const categoricalColumns = ref([]);
const categoricalToEncode = ref([]);
const uploadedFile = ref(null);
const triggerFileInput = () => {
  fileInput.value.click();
};

const handleFileUpload = async (event) => {
  const file = event.target.files[0];
  uploadedFile.value = file;

  if (file) {
    uploadedFileName.value = file.name;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("File upload failed");

      const data = await response.json();
      datasetPreview.value = data;

      selectedColumns.value = [...data.columns];
      categoricalColumns.value = data.categorical || [];
    } catch (error) {
      alert("Error al cargar el archivo: " + error.message);
    }
  }
};

const verifyIfLabelIsSelected = (col) => {
  return selectedColumns.value.includes(col);
};

const trainModel = () => {
  if (!datasetPreview.value || !selectedColumns.value.length) {
    console.error(
      "La vista previa del conjunto de datos o las columnas seleccionadas están incompletas"
    );
    return;
  }
  const preprocessingPayload = {
    missingValuesOption: missingValuesOption.value,
    selectedColumns: selectedColumns.value,
    categoricalColumns: categoricalToEncode.value,
    file: uploadedFile.value,
  };

  store.preprocessingPayload = preprocessingPayload;
  router.push({ name: "train-model" });
};
</script>

<template>
  <div class="min-h-screen bg-gray-50 flex flex-col items-center p-6">
    <div class="max-w-4xl w-full text-center mb-6">
      <h1 class="text-3xl font-bold text-gray-800">
        Algoritmo Supervisado: Random Forest
      </h1>
    </div>

    <div class="max-w-4xl w-full bg-white shadow-md rounded-lg p-6 mb-6">
      <div
        class="border-dashed border-2 border-gray-300 rounded-lg p-6 text-center text-gray-500"
      >
        <input
          type="file"
          accept=".csv, .xlsx"
          @change="handleFileUpload"
          class="hidden"
          ref="fileInput"
        />
        <button
          @click="triggerFileInput"
          class="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Seleccione Archivo
        </button>
      </div>
      <p v-if="uploadedFileName" class="text-sm text-gray-700 mt-4">
        Archivo seleccionado:
        <span class="font-medium">{{ uploadedFileName }}</span>
      </p>
    </div>

    <div
      v-if="datasetPreview"
      class="max-w-4xl w-full bg-white shadow-md rounded-lg p-6 mb-6"
    >
      <h2 class="text-lg font-semibold text-gray-700">
        Vista previa de los datos
      </h2>
      <div class="overflow-x-auto">
        <table class="table-auto w-full text-left text-sm text-gray-600">
          <thead>
            <tr class="bg-gray-200 text-gray-700">
              <th
                v-for="(col, index) in datasetPreview.columns"
                :key="index"
                class="px-4 py-2"
              >
                {{ col }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, rowIndex) in datasetPreview.rows" :key="rowIndex">
              <td
                v-for="(cell, cellIndex) in row"
                :key="cellIndex"
                class="border-t px-4 py-2"
              >
                {{ cell }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div
      v-if="datasetPreview"
      class="max-w-4xl w-full bg-white shadow-md rounded-lg p-6 mb-6"
    >
      <h2 class="text-lg font-semibold text-gray-700">
        Opciones de preprocesamiento
      </h2>
      <div class="space-y-4">
        <div>
          <label class="block text-gray-700 font-medium"
            >Manejar valores faltantes:</label
          >
          <select
            v-model="missingValuesOption"
            class="mt-2 block w-full border rounded px-3 py-2"
          >
            <option value="drop">Eliminar filas con valores faltantes</option>
            <option value="mean">Rellenar con media</option>
            <option value="median">Rellenar con mediana</option>
          </select>
        </div>

        <div>
          <label class="block text-gray-700 font-medium"
            >Seleccionar columnas para incluir:</label
          >
          <div class="mt-2 grid grid-cols-2 gap-2">
            <label
              v-for="(col, index) in datasetPreview.columns"
              :key="index"
              class="flex items-center"
            >
              <input
                type="checkbox"
                :value="col"
                v-model="selectedColumns"
                class="mr-2"
              />
              {{ col }}
            </label>
          </div>
        </div>

        <div>
          <label class="block text-gray-700 font-medium"
            >Codificar variables categóricas:</label
          >
          <div class="mt-2 grid grid-cols-2 gap-2">
            <label
              v-for="(col, index) in categoricalColumns"
              :key="index"
              class="flex items-center"
            >
              <input
                :disabled="!verifyIfLabelIsSelected(col)"
                type="checkbox"
                :value="col"
                v-model="categoricalToEncode"
                class="mr-2"
              />
              <span
                :class="{ 'text-slate-400': !verifyIfLabelIsSelected(col) }"
                >{{ col }}</span
              >
            </label>
          </div>
        </div>
      </div>
    </div>

    <div class="max-w-4xl w-full text-right">
      <button
        @click="trainModel"
        class="px-6 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        :disabled="!uploadedFileName"
        :class="{
          'bg-slate-500 hover:bg-slate-500 cursor-not-allowed':
            !uploadedFileName,
        }"
      >
        Siguiente: Entrenar Modelo
      </button>
    </div>
  </div>
</template>

<style scoped></style>
