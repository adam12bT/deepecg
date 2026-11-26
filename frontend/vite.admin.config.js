import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  root: fileURLToPath(new URL('./admin', import.meta.url)),
  base: '/admin-static/',
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:7880' } },
})
