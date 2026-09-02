/**
 * Virtual filesystem backing the browser Tauri shims.
 *
 * Paths the desktop app treats as real files ("/Users/…/graph.vn") become keys
 * in this store. Text entries persist in localStorage so autosave survives a
 * reload the way it does on desktop; binary blobs stay in memory only, since
 * images and videos would blow the storage quota.
 */

const STORAGE_PREFIX = 'vnstudio-vfs:'

const binaryFiles = new Map<string, { data: ArrayBuffer; mime: string; url: string }>()
const textFiles = new Map<string, string>()

function storageKey(path: string): string {
  return STORAGE_PREFIX + path
}

export function writeVirtualFile(path: string, contents: string): void {
  textFiles.set(path, contents)
  try {
    localStorage.setItem(storageKey(path), contents)
  } catch {
    // Quota exceeded — the in-memory copy still serves this session.
  }
}

export function readVirtualFile(path: string): string | null {
  const inMemory = textFiles.get(path)
  if (inMemory !== undefined) return inMemory
  try {
    return localStorage.getItem(storageKey(path))
  } catch {
    return null
  }
}

export function writeVirtualBinary(path: string, data: ArrayBuffer, mime: string): void {
  const previous = binaryFiles.get(path)
  if (previous) URL.revokeObjectURL(previous.url)
  const url = URL.createObjectURL(new Blob([data], { type: mime }))
  binaryFiles.set(path, { data, mime, url })
}

export function readVirtualBinary(path: string): ArrayBuffer | null {
  return binaryFiles.get(path)?.data ?? null
}

/**
 * Resolves a path to something an <img>/<video> can load: a blob URL for an
 * uploaded file, or the path itself when it points at a bundled asset.
 */
export function resolveMediaUrl(path: string): string {
  const entry = binaryFiles.get(path)
  if (entry) return entry.url
  const text = readVirtualFile(path)
  if (text !== null && text.startsWith('data:')) return text
  if (/^(https?:|data:|blob:)/.test(path)) return path
  // A bundled asset ("samples/foo.jpg") has to be resolved against the deploy
  // base, which is /<repo>/ on GitHub Pages rather than /.
  return import.meta.env.BASE_URL + path.replace(/^\//, '')
}

export function virtualFileExists(path: string): boolean {
  if (binaryFiles.has(path) || textFiles.has(path)) return true
  try {
    return localStorage.getItem(storageKey(path)) !== null
  } catch {
    return false
  }
}

export function listVirtualFiles(prefix: string): string[] {
  const names = new Set<string>()
  for (const key of textFiles.keys()) if (key.startsWith(prefix)) names.add(key)
  for (const key of binaryFiles.keys()) if (key.startsWith(prefix)) names.add(key)
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key?.startsWith(STORAGE_PREFIX)) {
        const path = key.slice(STORAGE_PREFIX.length)
        if (path.startsWith(prefix)) names.add(path)
      }
    }
  } catch {
    // localStorage unavailable — in-memory listing is all we have.
  }
  return [...names]
}

export function removeVirtualFile(path: string): void {
  const entry = binaryFiles.get(path)
  if (entry) URL.revokeObjectURL(entry.url)
  binaryFiles.delete(path)
  textFiles.delete(path)
  try {
    localStorage.removeItem(storageKey(path))
  } catch {
    // Nothing to clean up.
  }
}

/** Triggers a real browser download — how "saving to disk" surfaces on the web. */
export function downloadFile(filename: string, contents: string | ArrayBuffer, mime: string): void {
  const blob = contents instanceof ArrayBuffer ? new Blob([contents], { type: mime }) : new Blob([contents], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename.split('/').pop() || 'download'
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
