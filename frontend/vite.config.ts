import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Hardcoding the API URL to ensure it uses the correct port
const API_URL = 'http://localhost:8008'
console.log(`Using API URL (hardcoded): ${API_URL}`)

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: API_URL,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  define: {
    // Make API URL available globally in the app
    '__API_URL__': JSON.stringify(API_URL)
  },
  resolve: {
    alias: {
      '@': '/src'
    }
  }
})
