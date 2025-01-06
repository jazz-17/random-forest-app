import { defineStore } from 'pinia'
export const modelStore = defineStore('model', {
    state: () => ({
        preprocessingPayload: null,
        modelId: null,
    }),
    actions: {
        increment() {
            this.count++
        },
        setModel(model) {
            this.model = model
        },
        setResults(results) {
            this.results = results
        },
    },
})
