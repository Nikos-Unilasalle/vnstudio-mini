/**
 * Browser stand-in for @tauri-apps/api/plugin-opener.
 *
 * "Reveal this file in Finder" has no web equivalent; only http(s) links can be
 * meaningfully opened, and a folder path is silently ignored.
 */
export async function openPath(path: string): Promise<void> {
  if (/^https?:\/\//.test(path)) window.open(path, '_blank', 'noopener')
}

export async function openUrl(url: string): Promise<void> {
  window.open(url, '_blank', 'noopener')
}
