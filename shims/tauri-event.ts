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

function mimeFor(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  switch (ext) {
    case 'png': return 'image/png'
    case 'jpg':
    case 'jpeg': return 'image/jpeg'
    case 'webp': return 'image/webp'
    case 'bmp': return 'image/bmp'
    case 'gif': return 'image/gif'
    case 'tif':
    case 'tiff': return 'image/tiff'
    case 'mp4': return 'video/mp4'
    case 'webm': return 'video/webm'
    case 'mov': return 'video/quicktime'
    case 'avi': return 'video/x-msvideo'
    case 'mkv': return 'video/x-matroska'
    default: return 'application/octet-stream'
  }
}

async function ingest(file: File): Promise<string> {
  const path = `vfs:/${file.name}`
  ;(file as any).path = path
  const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
  if (TEXT_EXTENSIONS.has(ext)) writeVirtualFile(path, await file.text())
  else writeVirtualBinary(path, await file.arrayBuffer(), file.type || mimeFor(file.name))
  return path
}

export async function listen<T = unknown>(event: string, handler: EventCallback<T>): Promise<() => void> {
  if (event !== 'tauri://drag-drop') return () => {}

  const onDragOver = (e: DragEvent) => e.preventDefault()
  const onDrop = async (e: DragEvent) => {
    if (!e.dataTransfer?.files?.length) return
    const paths = await Promise.all(Array.from(e.dataTransfer.files).map(ingest))
    const target = e.target as HTMLElement | null
    if (target?.closest('.react-flow__node')) {
      return
    }
    e.preventDefault()
    handler({ payload: { paths, position: { x: e.clientX, y: e.clientY } } as T })
  }

  window.addEventListener('dragover', onDragOver)
  window.addEventListener('drop', onDrop, true)
  return () => {
    window.removeEventListener('dragover', onDragOver)
    window.removeEventListener('drop', onDrop, true)
  }
}

export async function emit(_event: string, _payload?: unknown): Promise<void> {}
