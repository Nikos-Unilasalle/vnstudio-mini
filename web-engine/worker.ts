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

export type WorkerRequest =
  | { type: 'init' }
  | {
      type: 'run'
      requestId: number
      nodes: GraphNode[]
      edges: GraphEdge[]
      previewNodeId: string | null
      frames: Record<string, CapturedFrame>
    }

export type WorkerResponse =
  | { type: 'schemas'; schemas: typeof SCHEMAS }
  | { type: 'progress'; progress: number | null; message: string }
  | { type: 'ready' }
  | { type: 'load-error'; message: string }
  | { type: 'result'; requestId: number; nodesData: Record<string, unknown>; frame: string | null; errors: Record<string, string> }
  | { type: 'run-error'; requestId: number; message: string }
  // A node asked to save a file; only the main thread has the DOM to do it.
  | { type: 'download'; filename: string; contents: string | ArrayBuffer; mime: string }

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

  try {
    const result = await executor.run(message.nodes, message.edges, message.previewNodeId, message.frames)
    try {
      postMessage({ type: 'result', requestId: message.requestId, ...result } satisfies WorkerResponse)
    } catch {
      // Something in nodesData refused to clone. The executor screens for that,
      // but a node returning an exotic value must degrade to a frame without
      // live data rather than killing the run.
      postMessage({
        type: 'result',
        requestId: message.requestId,
        nodesData: {},
        frame: result.frame,
        errors: result.errors,
      } satisfies WorkerResponse)
    }
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error)
    postMessage({ type: 'run-error', requestId: message.requestId, message: text } satisfies WorkerResponse)
  }
}
