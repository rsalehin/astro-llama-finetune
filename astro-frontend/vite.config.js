import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite' // This was the missing part!

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
})