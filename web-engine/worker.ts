/**
 * Dedicated worker hosting OpenCV.js and the graph executor.
 *
 * Everything CPU-heavy — OpenCV's WASM init and every node's Mat processing —
 * happens here instead of on the main thread, so the page stays interactive
 * no matter how long OpenCV takes to initialise or a frame takes to process
 * (see opencv.ts for why that init is unavoidably synchronous). Spawned once
 * by shims/useVisionEngine.ts.
 */
import './importScriptsPolyfill'
import { loadOpenCv } from './opencv'
import { GraphExecutor, type GraphEdge, type GraphNode } from './executor'
import { SCHEMAS } from './registry'
import type { CapturedFrame } from './types'
import { setTextFiles } from './textFiles'

export type WorkerRequest =
  | { type: 'init' }
  | {
      type: 'run'
      requestId: number
      nodes: GraphNode[]
      edges: GraphEdge[]
      previewNodeId: string | null
      frames: Record<string, CapturedFrame>
      /** Only present when the store changed since the last run. */
      textFiles?: Record<string, string>
    }

export type WorkerResponse =
  | { type: 'schemas'; schemas: typeof SCHEMAS }
  | { type: 'progress'; progress: number | null; message: string }
  | { type: 'ready' }
  | { type: 'load-error'; message: string }
  | {
      type: 'result'
      requestId: number
      nodesData: Record<string, unknown>
      /** Transferred, not copied; the main thread owns it and must close() it. */
      frameBitmap: ImageBitmap | null
      frame: string | null
      errors: Record<string, string>
    }
  | { type: 'run-error'; requestId: number; message: string }
  // A node asked to save a file; only the main thread has the DOM to do it.
  | { type: 'download'; filename: string; contents: string | ArrayBuffer; mime: string }

/**
 * `postMessage` with a transfer list.
 *
 * The ambient types here resolve the DOM's Window.postMessage rather than the
 * worker's, whose second argument is the list of objects to move instead of
 * copy — which is the whole point of sending the preview as an ImageBitmap.
 */
function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  ;(self as unknown as { postMessage: (m: WorkerResponse, t: Transferable[]) => void }).postMessage(message, transfer)
}

let executor: GraphExecutor | null = null

postMessage({ type: 'schemas', schemas: SCHEMAS } satisfies WorkerResponse)

loadOpenCv((progress) => {
  postMessage({ type: 'progress', progress: progress.progress, message: progress.message } satisfies WorkerResponse)
})
  .then(({ cv }) => {
    executor = new GraphExecutor(cv)
    postMessage({ type: 'ready' } satisfies WorkerResponse)
  })
  .catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error)
    postMessage({ type: 'load-error', message } satisfies WorkerResponse)
  })

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data
  if (message.type !== 'run') return

  if (!executor) {
    postMessage({ type: 'run-error', requestId: message.requestId, message: 'OpenCV n’est pas encore prêt.' } satisfies WorkerResponse)
    return
  }

  // The snapshot arrives only when it has changed, so an unchanged store is
  // simply left in place rather than re-sent with every frame.
  if (message.textFiles) setTextFiles(message.textFiles)

  try {
    let result
    try {
      result = await executor.run(message.nodes, message.edges, message.previewNodeId, message.frames)
    } finally {
      // The captured frames were transferred in, so this worker owns them and
      // nothing else will free them. An ImageBitmap holds its pixels outside
      // the JavaScript heap, where the collector has no reason to hurry, so a
      // webcam graph running thirty times a second was stacking up megabytes a
      // second — including on the runs that fail, which is why this releases
      // them whatever happened.
      for (const frame of Object.values(message.frames ?? {})) frame.bitmap?.close()
    }

    const transfer = result.frameBitmap ? [result.frameBitmap] : []
    try {
      post({ type: 'result', requestId: message.requestId, ...result }, transfer)
    } catch {
      // Something in nodesData refused to clone. The executor screens for that,
      // but a node returning an exotic value must degrade to a frame without
      // live data rather than killing the run. The bitmap is gone either way:
      // a failed postMessage still detaches whatever was in the transfer list.
      post({
        type: 'result',
        requestId: message.requestId,
        nodesData: {},
        frameBitmap: null,
        frame: result.frame,
        errors: result.errors,
      })
    }
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error)
    postMessage({ type: 'run-error', requestId: message.requestId, message: text } satisfies WorkerResponse)
  }
}
