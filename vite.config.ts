import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Project pages are served from /<repo-name>/ — override via VITE_BASE if the repo name differs.
// Dev server stays at "/" for a simpler local preview.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? (process.env.VITE_BASE ?? '/vnstudio-mini/') : '/',
  plugins: [react()],
}))
