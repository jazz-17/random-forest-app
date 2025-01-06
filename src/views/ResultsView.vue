<template>
    <div class="min-h-screen bg-gray-50 flex flex-col items-center p-6">
        <div class="max-w-4xl w-full text-center mb-6">
            <h1 class="text-3xl font-bold text-gray-800">Heart Disease Prediction</h1>
        </div>

        <!-- Input Form -->
        <div class="max-w-4xl w-full bg-white shadow-md rounded-lg p-6">
            <h2 class="text-lg font-semibold text-gray-700 mb-4">Enter Patient Data</h2>
            <div class="space-y-4">
                <template v-for="(value, key) in patientData" class="flex flex-col">
                    <div v-if="categoricalColumns.includes(key)" class="flex flex-col">
                        <label>{{ key }}</label>
                        <input type="text" class="bg-gray-50 p-1 border" v-model="patientData[key]">
                        </input>
                    </div>
                    <div v-else class="flex flex-col">
                        <label>{{ key }}</label>
                        <input type="number" class="bg-gray-50 p-1 border" v-model="patientData[key]" />
                    </div>
                </template>
            </div>
        </div>

        <!-- Action Buttons -->
        <div class="max-w-4xl w-full text-right mt-4">
            <button @click="predictDisease" :disabled="isPredicting" :class="{
                'bg-slate-500 hover:bg-slate-500 cursor-not-allowed': isPredicting,
                'bg-blue-500 hover:bg-blue-600': !isPredicting,
            }" class="px-6 py-2 text-white rounded">
                Predict
            </button>
        </div>

        <!-- Loading Indicator -->
        <div v-if="isPredicting" class="flex items-center justify-center mt-6">
            <div class="spinner-border animate-spin inline-block w-8 h-8 border-4 rounded-full text-blue-500"></div>
            <p class="ml-4 text-gray-700">Making predictions...</p>
        </div>

        <!-- Prediction Results -->
        <div v-if="result" class="max-w-4xl w-full bg-white shadow-md rounded-lg p-6 mt-6">
            <h2 class="text-lg font-semibold text-gray-700 mb-4">Prediction Results</h2>
            <p class="text-gray-800">
                <strong>Prediction:</strong> {{ result.predictions[0] === 1 ? "Heart Disease Detected" : "No Heart Disease" }}
            </p>
            <div v-if="result.probabilities" class="mt-4">
                <h3 class="text-gray-700 font-medium">Class Probabilities:</h3>
                <ul class="list-disc pl-5">
                    <li v-for="(prob, index) in result.probabilities[0]" :key="index">
                        Class {{ index }}: {{ (prob * 100).toFixed(2) }}%
                    </li>
                </ul>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref } from "vue";
import { useRoute } from "vue-router";
import { modelStore } from "../store/modelStore";
const store = modelStore();
const route = useRoute();
const categoricalColumns = store.preprocessingPayload.categoricalColumns;

const patientData = store.preprocessingPayload.selectedColumns.reduce((acc, col) => {
    acc[col] = null;
    return acc;
}, {});
const modelPath = store.modelId;
// State management
const isPredicting = ref(false);
const result = ref(null);

const predictDisease = async () => {
    isPredicting.value = true;
    result.value = null;

    try {
        if (!modelPath || !patientData) {
            alert("Model path or patient data is missing.");
            isPredicting.value = false;
            return;
        }

        // Create FormData object
        const formData = new FormData();
        formData.append("modelPath", modelPath);
        formData.append("inputData", JSON.stringify(patientData)); // Serialize inputData as JSON

        // Send POST request
        const response = await fetch("/api/test", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Prediction failed: ${response.statusText}`);
        }

        // Handle response
        const data = await response.json();
        result.value = data;
    } catch (error) {
        console.error("Error during prediction:", error);
        alert("Failed to make predictions. Please try again.");
    } finally {
        isPredicting.value = false;
    }
};



</script>

<style scoped>
.spinner-border {
    border: 4px solid transparent;
    border-top-color: currentColor;
    border-right-color: currentColor;
}
</style>