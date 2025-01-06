<template>
  <div class="min-h-screen bg-gray-50 flex flex-col items-center p-6">
    <div class="max-w-4xl w-full text-center mb-6">
      <h1 class="text-3xl font-bold text-gray-800">Modelo Random Forest</h1>
    </div>

    <div class="max-w-4xl w-full bg-white shadow-md rounded-lg p-6 mb-6">
      <h2 class="text-lg font-semibold text-gray-700 mb-8">Hiperparámetros</h2>
      <div class="space-y-4">

        <div>
          <label class="block text-gray-700 font-medium">Número de árboles:</label>
          <input type="range" min="10" max="500" step="10" v-model="numTrees" class="mt-2 w-full" />
          <p class="text-gray-600 text-sm mt-1">Seleccionado: {{ numTrees }}</p>
        </div>

        <div>
          <label class="block text-gray-700 font-medium">Profundidad máxima:</label>
          <input type="range" min="1" max="50" step="1" v-model="maxDepth" class="mt-2 w-full" />
          <p class="text-gray-600 text-sm mt-1">Seleccionado: {{ maxDepth }}</p>
        </div>
      </div>
    </div>


    <div class="max-w-4xl w-full text-right">
      <button @click="startTraining" :disabled="isTraining"
        :class="{ 'bg-slate-500 hover:bg-slate-500 cursor-not-allowed': isTraining, 'bg-green-500 hover:bg-green-600': !isTraining }"
        class="px-6 py-2 text-white rounded">
        Comenzar entrenamiento
      </button>
    </div>

    <div v-if="isTraining" class="flex items-center justify-center mt-6">
      <div class="spinner-border animate-spin inline-block w-8 h-8 border-4 rounded-full text-blue-500"></div>
      <p class="ml-4 text-gray-700">Entrenamiento en progreso... </p>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import {modelStore} from "../store/modelStore"

const store = modelStore();
const preprocessingPayload = store.preprocessingPayload;
const router = useRouter();
const route = useRoute();

const numTrees = ref(100);
const maxDepth = ref(10);
const isTraining = ref(false);

const startTraining = async () => {
  isTraining.value = true;

  try {
    if (!preprocessingPayload || !preprocessingPayload.file) {
      throw new Error("Faltan datos");
    }

    const formData = new FormData();
    formData.append("numTrees", numTrees.value);
    formData.append("maxDepth", maxDepth.value);
    formData.append("missingValuesOption", JSON.stringify(preprocessingPayload.missingValuesOption));
    formData.append("selectedColumns", JSON.stringify(preprocessingPayload.selectedColumns));
    formData.append("categoricalToEncode", JSON.stringify(preprocessingPayload.categoricalToEncode));
    formData.append("file", preprocessingPayload.file);

    const response = await fetch("/api/train", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Training failed: ${response.statusText}`);
    }

    const result = await response.json();
    store.modelId = result.modelId;
    router.push({
      name: "results",
    });
  } catch (error) {
    console.error("Error al entrenar modelo:", error);
  } finally {
    isTraining.value = false;
  }
}
</script>

<style scoped>
.spinner-border {
  border: 4px solid transparent;
  border-top-color: currentColor;
  border-right-color: currentColor;
}
</style>
