/**
 * Browser stand-in for @tauri-apps/api/core.
 *
 * Rust commands have no browser equivalent. Every `invoke` call site in the app
 * already tolerates a rejected promise (VNPad pairing, schema push), so failing
 * loudly here would only add noise — the call resolves to null instead.
 */
export async function invoke<T = unknown>(_cmd: string, _args?: Record<string, unknown>): Promise<T> {
  return null as T
}

export function convertFileSrc(filePath: string): string {
  return filePath
}
