import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'file-upload',
      component: () => import('../views/UploadView.vue'),
    },
    {
      path: '/train-model',
      name: 'train-model',
      component: () => import('../views/TrainingView.vue'),
      props: true,
    },
    {
      path: '/results',
      name: 'results',
      component: () => import('../views/ResultsView.vue'),
      props: true,
    }
  ],
})

export default router
