/**
 * The Worker's copy of the text files the user has loaded.
 *
 * Node implementations run in the graph Worker, which cannot reach the main
 * thread's virtual filesystem or localStorage. The hook ships a snapshot with
 * the run request whenever it changes, and this holds it for nodes that read a
 * path — the CSV Reader above all.
 */
let files: Record<string, string> = {}

export function setTextFiles(next: Record<string, string>): void {
  files = next
}

/** The file's contents, or null when nothing has been loaded under that path. */
export function readTextFile(path: string): string | null {
  if (path in files) return files[path]
  // A path typed without the vfs: prefix should still find a dropped file.
  const bare = path.replace(/^vfs:\//, '')
  for (const key of Object.keys(files)) {
    if (key === `vfs:/${bare}` || key.endsWith(`/${bare}`)) return files[key]
  }
  return null
}

/** Every known path, for a node that wants to offer a choice. */
export function listTextFiles(): string[] {
  return Object.keys(files)
}
