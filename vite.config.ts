import { fileURLToPath, URL } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const resolvePath = (relative: string) => fileURLToPath(new URL(relative, import.meta.url))

/**
 * `src/` is a verbatim copy of the desktop VNStudio frontend and must stay that
 * way, so every difference the web build needs is expressed here as an alias:
 *
 *  - the Tauri plugin APIs resolve to browser shims (file dialogs, virtual FS,
 *    drag-drop, fullscreen) instead of failing to resolve at all;
 *  - `useVisionEngine` — the one module that talks to the Python engine over a
 *    WebSocket — resolves to an in-page OpenCV.js executor with the same
 *    signature.
 *
 * Anything else in the UI is byte-identical to the desktop app.
 */
/**
 * The desktop app fetches bundled templates from the absolute path
 * `/templates/…`, which is correct when it serves from the root. GitHub Pages
 * serves a project from `/<repo>/`, so those requests would 404. Rewriting the
 * literal at transform time keeps src/ verbatim while making the URLs
 * base-aware.
 */
function baseAwareTemplateUrls() {
  return {
    name: 'base-aware-template-urls',
    transform(code: string, id: string) {
      if (!id.includes('/src/')) return null
      const rewritten = code
        // fetch('/templates/manifest.json')
        .replace(/'\/templates\//g, "import.meta.env.BASE_URL + 'templates/")
        // fetch(`/templates/${file}`)
        .replace(/`\/templates\//g, '`${import.meta.env.BASE_URL}templates/')
      return rewritten === code ? null : { code: rewritten, map: null }
    },
  }
}

export default defineConfig(({ command }) => ({
  // GitHub Pages serves the project from /<repo>/; dev stays at the root.
  base: command === 'build' ? (process.env.VITE_BASE ?? '/vnstudio-mini/') : '/',
  plugins: [react(), baseAwareTemplateUrls()],
  resolve: {
    alias: [
      { find: '@tauri-apps/plugin-dialog', replacement: resolvePath('./shims/tauri-dialog.ts') },
      { find: '@tauri-apps/plugin-fs', replacement: resolvePath('./shims/tauri-fs.ts') },
      { find: '@tauri-apps/plugin-opener', replacement: resolvePath('./shims/tauri-opener.ts') },
      { find: '@tauri-apps/api/core', replacement: resolvePath('./shims/tauri-core.ts') },
      { find: '@tauri-apps/api/event', replacement: resolvePath('./shims/tauri-event.ts') },
      { find: '@tauri-apps/api/path', replacement: resolvePath('./shims/tauri-path.ts') },
      { find: '@tauri-apps/api/window', replacement: resolvePath('./shims/tauri-window.ts') },
      { find: '@tauri-apps/api/webviewWindow', replacement: resolvePath('./shims/tauri-webview-window.ts') },
      // Exact match only: the shim itself imports from src/, and a prefix rule
      // would rewrite that import back onto the shim.
      { find: /^.*\/hooks\/useVisionEngine$/, replacement: resolvePath('./shims/useVisionEngine.ts') },
    ],
  },
  optimizeDeps: {
    // Pulled from a CDN at runtime, never bundled.
    exclude: ['@mediapipe/tasks-vision'],
  },
}))
