/**
 * Browser stand-in for @tauri-apps/api/webviewWindow — the popout preview
 * becomes a real popup window instead of a second webview.
 */
interface WebviewWindowOptions {
  url?: string
  title?: string
  width?: number
  height?: number
  minWidth?: number
  minHeight?: number
  resizable?: boolean
  [key: string]: unknown
}

export class WebviewWindow {
  private handle: Window | null

  constructor(_label: string, options: WebviewWindowOptions = {}) {
    const { url = '', title = 'VisionNodes', width = 800, height = 600 } = options
    this.handle = window.open(url, title, `width=${width},height=${height}`)
  }

  async once(_event: string, handler: (event?: unknown) => void): Promise<void> {
    handler({ payload: null })
  }

  async emit(_event: string, _payload?: unknown): Promise<void> {}
  async close(): Promise<void> {
    this.handle?.close()
  }
}
