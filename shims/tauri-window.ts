/**
 * Browser stand-in for @tauri-apps/api/window — maps the fullscreen toggle onto
 * the Fullscreen API.
 */
export function getCurrentWindow() {
  return {
    async isFullscreen(): Promise<boolean> {
      return document.fullscreenElement !== null
    },
    async setFullscreen(value: boolean): Promise<void> {
      if (value) await document.documentElement.requestFullscreen().catch(() => {})
      else if (document.fullscreenElement) await document.exitFullscreen().catch(() => {})
    },
    async setTitle(title: string): Promise<void> {
      document.title = title
    },
    async close(): Promise<void> {},
  }
}
