/**
 * Browser stand-in for @tauri-apps/api/event.
 *
 * The only listener the app registers is 'tauri://drag-drop', which the desktop
 * shell fires with OS file paths. The browser has its own drag-and-drop events
 * on the window, so this bridges them into the same callback shape.
 */
import { writeVirtualBinary, writeVirtualFile } from './vfs'

type EventCallback<T = unknown> = (event: { payload: T }) => void

const TEXT_EXTENSIONS = new Set(['vn', 'json', 'csv', 'txt', 'md', 'py', 'toml', 'geojson', 'svg'])

async function ingest(file: File): Promise<string> {
  const path = `vfs:/${file.name}`
  const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
  if (TEXT_EXTENSIONS.has(ext)) writeVirtualFile(path, await file.text())
  else writeVirtualBinary(path, await file.arrayBuffer(), file.type)
  return path
}

export async function listen<T = unknown>(event: string, handler: EventCallback<T>): Promise<() => void> {
  if (event !== 'tauri://drag-drop') return () => {}

  const onDragOver = (e: DragEvent) => e.preventDefault()
  const onDrop = async (e: DragEvent) => {
    if (!e.dataTransfer?.files?.length) return
    e.preventDefault()
    const paths = await Promise.all(Array.from(e.dataTransfer.files).map(ingest))
    handler({ payload: { paths, position: { x: e.clientX, y: e.clientY } } as T })
  }

  window.addEventListener('dragover', onDragOver)
  window.addEventListener('drop', onDrop)
  return () => {
    window.removeEventListener('dragover', onDragOver)
    window.removeEventListener('drop', onDrop)
  }
}

export async function emit(_event: string, _payload?: unknown): Promise<void> {}
