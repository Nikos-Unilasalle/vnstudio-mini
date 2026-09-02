/**
 * Browser stand-in for @tauri-apps/plugin-fs.
 *
 * Reads resolve against the virtual filesystem (see vfs.ts). Writes split by
 * intent: a path the user chose through a save dialog becomes a real browser
 * download, while background writes — autosave, API keys — stay in the virtual
 * store so they persist across reloads without prompting anyone.
 */
import {
  downloadFile,
  listVirtualFiles,
  readVirtualBinary,
  readVirtualFile,
  virtualFileExists,
  writeVirtualBinary,
  writeVirtualFile,
} from './vfs'

interface FsOptions {
  baseDir?: unknown
  recursive?: boolean
}

/** Paths under these prefixes are app state, not user documents — never download them. */
const INTERNAL_PREFIXES = ['.vnstudio', 'vnstudio-autosave', '.config']

function isInternal(path: string): boolean {
  return INTERNAL_PREFIXES.some((prefix) => path.startsWith(prefix) || path.includes(`/${prefix}`))
}

function mimeFor(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase()
  if (ext === 'json' || ext === 'vn') return 'application/json'
  if (ext === 'csv') return 'text/csv'
  if (ext === 'svg') return 'image/svg+xml'
  if (ext === 'png') return 'image/png'
  if (ext === 'py') return 'text/x-python'
  return 'text/plain'
}

export async function readTextFile(path: string, _options?: FsOptions): Promise<string> {
  const contents = readVirtualFile(path)
  if (contents !== null) return contents
  throw new Error(`File not found: ${path}`)
}

export async function writeTextFile(path: string, contents: string, _options?: FsOptions): Promise<void> {
  writeVirtualFile(path, contents)
  if (!isInternal(path)) downloadFile(path, contents, mimeFor(path))
}

export async function readFile(path: string, _options?: FsOptions): Promise<Uint8Array> {
  const buffer = readVirtualBinary(path)
  if (buffer) return new Uint8Array(buffer)
  const text = readVirtualFile(path)
  if (text !== null) return new TextEncoder().encode(text)
  throw new Error(`File not found: ${path}`)
}

export async function writeFile(path: string, data: Uint8Array, _options?: FsOptions): Promise<void> {
  const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer
  writeVirtualBinary(path, buffer, mimeFor(path))
  if (!isInternal(path)) downloadFile(path, buffer, mimeFor(path))
}

export async function exists(path: string, _options?: FsOptions): Promise<boolean> {
  return virtualFileExists(path)
}

export async function mkdir(_path: string, _options?: FsOptions): Promise<void> {
  // Directories are implicit in a flat key-value store.
}

export async function remove(path: string, _options?: FsOptions): Promise<void> {
  const { removeVirtualFile } = await import('./vfs')
  removeVirtualFile(path)
}

export async function rename(oldPath: string, newPath: string, _options?: FsOptions): Promise<void> {
  const text = readVirtualFile(oldPath)
  if (text !== null) {
    writeVirtualFile(newPath, text)
    const { removeVirtualFile } = await import('./vfs')
    removeVirtualFile(oldPath)
  }
}

export interface DirEntry {
  name: string
  isFile: boolean
  isDirectory: boolean
  isSymlink: boolean
}

export async function readDir(path: string, _options?: FsOptions): Promise<DirEntry[]> {
  const prefix = path.endsWith('/') ? path : `${path}/`
  return listVirtualFiles(prefix).map((full) => ({
    name: full.slice(prefix.length),
    isFile: true,
    isDirectory: false,
    isSymlink: false,
  }))
}

export const BaseDirectory = {
  Home: 1,
  AppConfig: 2,
  AppData: 3,
  Document: 4,
  Download: 5,
} as const
