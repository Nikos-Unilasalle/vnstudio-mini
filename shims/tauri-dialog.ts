/**
 * Browser stand-in for @tauri-apps/plugin-dialog.
 *
 * The desktop app opens native file dialogs and passes real filesystem paths
 * around. In the browser there are no paths, so `open()` returns a synthetic
 * "vfs:" path and stashes the file's contents in the virtual filesystem that
 * tauri-fs.ts reads back. Downstream code keeps treating the return value as an
 * opaque path string, which is all it ever did.
 */
import { writeVirtualFile, writeVirtualBinary } from './vfs'

interface DialogFilter {
  name: string
  extensions: string[]
}

interface OpenOptions {
  multiple?: boolean
  directory?: boolean
  filters?: DialogFilter[]
  defaultPath?: string
  title?: string
}

interface SaveOptions {
  filters?: DialogFilter[]
  defaultPath?: string
  title?: string
}

const TEXT_EXTENSIONS = new Set(['vn', 'json', 'csv', 'txt', 'md', 'py', 'toml', 'yaml', 'yml', 'geojson', 'svg'])

function acceptFromFilters(filters?: DialogFilter[]): string {
  if (!filters?.length) return ''
  return filters
    .flatMap((f) => f.extensions)
    .filter((e) => e && e !== '*')
    .map((e) => `.${e}`)
    .join(',')
}

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

function pickFiles(accept: string, multiple: boolean): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = accept
    input.multiple = multiple
    // Safari / WebKit ignores .click() on elements with display: none.
    // Offscreen fixed positioning hides the element visually without breaking programmatic click.
    input.style.position = 'fixed'
    input.style.top = '-1000px'
    input.style.left = '-1000px'
    input.style.opacity = '0'
    input.style.pointerEvents = 'none'

    let settled = false
    const done = (files: File[]) => {
      if (settled) return
      settled = true
      input.remove()
      resolve(files)
    }
    input.addEventListener('change', () => done(input.files ? Array.from(input.files) : []))
    input.addEventListener('cancel', () => done([]))
    window.addEventListener(
      'focus',
      () => {
        setTimeout(() => {
          if (!settled) done([])
        }, 1000)
      },
      { once: true }
    )
    document.body.appendChild(input)
    input.click()
  })
}

async function ingest(file: File): Promise<string> {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
  const path = `vfs:/${file.name}`
  ;(file as any).path = path
  if (TEXT_EXTENSIONS.has(ext)) {
    writeVirtualFile(path, await file.text())
  } else {
    writeVirtualBinary(path, await file.arrayBuffer(), file.type || mimeFor(file.name))
  }
  return path
}

export async function open(options: OpenOptions = {}): Promise<string | string[] | null> {
  if (options.directory) {
    // No browser equivalent of "pick a folder and hand me its path".
    return null
  }
  const files = await pickFiles(acceptFromFilters(options.filters), !!options.multiple)
  if (files.length === 0) return null
  const paths = await Promise.all(files.map(ingest))
  return options.multiple ? paths : paths[0]
}

export async function save(options: SaveOptions = {}): Promise<string | null> {
  // There is no "choose where to save" step in the browser — the download lands
  // in the user's Downloads folder when tauri-fs.ts writes to this path.
  const suggested = options.defaultPath?.split('/').pop()
  const ext = options.filters?.[0]?.extensions?.[0]
  const fallback = ext ? `export.${ext}` : 'export'
  return `vfs:/${suggested || fallback}`
}

interface MessageOptions {
  title?: string
  kind?: 'info' | 'warning' | 'error'
  okLabel?: string
  cancelLabel?: string
  [key: string]: unknown
}

export async function ask(message: string, options?: MessageOptions | string): Promise<boolean> {
  const title = typeof options === 'string' ? options : options?.title
  return window.confirm(title ? `${title}\n\n${message}` : message)
}

export async function message(msg: string, options?: MessageOptions | string): Promise<void> {
  const title = typeof options === 'string' ? options : options?.title
  window.alert(title ? `${title}\n\n${msg}` : msg)
}

export async function confirm(msg: string, options?: MessageOptions | string): Promise<boolean> {
  return ask(msg, options)
}
