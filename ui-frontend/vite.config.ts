import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../agentcost/ui/static',
    emptyOutDir: true,
  },
})
