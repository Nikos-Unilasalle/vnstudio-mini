/**
 * Drop-in replacement for the desktop `useVisionEngine`.
 *
 * The desktop hook streams the graph to a Python process over a WebSocket and
 * receives rendered frames back. This one keeps the identical return shape but
 * evaluates the graph with OpenCV.js in a dedicated Worker (see
 * web-engine/worker.ts) — OpenCV's WASM init and every node's Mat processing
 * happen off the main thread, so the page stays interactive regardless of how
 * long that takes. Every consumer in src/ is untouched.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createNodesDataStore } from '../src/context/NodesDataContext'
import { resolveMediaUrl } from './vfs'
import { MediaFrameSource } from './mediaFrameSource'
import type { WorkerRequest, WorkerResponse } from '../web-engine/worker'
import type { GraphEdge, GraphNode } from '../web-engine/executor'
import type { CapturedFrame } from '../web-engine/types'

export type EngineNotification = {
  id: string
  message: string
  progress: number | null
  level: 'info' | 'warning' | 'error'
}

/** Coalesces the bursts of graph updates React emits while a slider is dragged. */
const RUN_DEBOUNCE_MS = 60

/** input_image's `path` param has to be resolved to a fetchable URL before the
 * worker sees it: the blob-URL ↔ path mapping that resolveMediaUrl reads
 * lives in this module's main-thread memory (see shims/vfs.ts), which the
 * worker — a separate JS realm — doesn't share. */
function resolveInputPaths(nodes: GraphNode[]): GraphNode[] {
  return nodes.map((node) => {
    if (node.type !== 'input_image') return node
    const path = node.data?.params?.path
    if (typeof path !== 'string' || !path) return node
    return { ...node, data: { ...node.data, params: { ...node.data?.params, path: resolveMediaUrl(path) } } }
  })
}

export function useVisionEngine(onCapture?: (nodeId: string, base64: string) => void) {
  const [frame, setFrame] = useState<string | null>(null)
  const nodesDataStore = useMemo(() => createNodesDataStore(), [])
  const nodesDataRef = useRef<Record<string, any>>({})
  const [pluginSchemas, setPluginSchemas] = useState<any[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [notifications, setNotifications] = useState<EngineNotification[]>([])
  const [computingNodeId, setComputingNodeId] = useState<string | null>(null)

  const workerRef = useRef<Worker | null>(null)
  const mediaRef = useRef<MediaFrameSource>(new MediaFrameSource())
  const pendingRuns = useRef(
    new Map<number, { resolve: (r: { nodesData: Record<string, unknown>; frame: string | null; errors: Record<string, string> }) => void; reject: (e: Error) => void }>()
  )
  const nextRequestId = useRef(1)

  const graphRef = useRef<{ nodes: GraphNode[]; edges: GraphEdge[] }>({ nodes: [], edges: [] })
  const previewNodeRef = useRef<string | null>(null)
  const dismissTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})
  const runTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isRunning = useRef(false)
  const runQueued = useRef(false)
  const captureRequests = useRef<Set<string>>(new Set())

  const dismissNotification = useCallback((id: string) => {
    clearTimeout(dismissTimers.current[id])
    delete dismissTimers.current[id]
    setNotifications((prev) => prev.filter((n) => n.id !== id))
  }, [])

  const pushNotification = useCallback(
    (message: string, level: EngineNotification['level'] = 'info', ttl = 4000) => {
      const id = `fe_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
      setNotifications((prev) => [...prev.slice(-9), { id, message, progress: null, level }])
      if (ttl > 0) dismissTimers.current[id] = setTimeout(() => dismissNotification(id), ttl)
    },
    [dismissNotification]
  )

  useEffect(() => {
    let cancelled = false
    const LOADING_ID = 'opencv_loading'

    const worker = new Worker(new URL('../web-engine/worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      if (cancelled) return
      const message = event.data
      switch (message.type) {
        case 'schemas':
          setPluginSchemas(message.schemas)
          break
        case 'progress':
          setNotifications((prev) => {
            const entry: EngineNotification = { id: LOADING_ID, message: message.message, progress: message.progress, level: 'info' }
            const index = prev.findIndex((n) => n.id === LOADING_ID)
            return index >= 0 ? prev.map((n, i) => (i === index ? entry : n)) : [...prev, entry]
          })
          break
        case 'ready':
          setIsConnected(true)
          setNotifications((prev) => prev.filter((n) => n.id !== LOADING_ID))
          pushNotification('Moteur navigateur prêt ✓', 'info', 2500)
          break
        case 'load-error':
          setNotifications((prev) =>
            prev.map((n) => (n.id === LOADING_ID ? { ...n, message: message.message, progress: null, level: 'error' } : n))
          )
          break
        case 'result': {
          const pending = pendingRuns.current.get(message.requestId)
          pending?.resolve(message)
          pendingRuns.current.delete(message.requestId)
          break
        }
        case 'run-error': {
          const pending = pendingRuns.current.get(message.requestId)
          pending?.reject(new Error(message.message))
          pendingRuns.current.delete(message.requestId)
          break
        }
        case 'download': {
          // The worker has no DOM, so export nodes send the bytes here instead.
          const blob = new Blob([message.contents], { type: message.mime })
          const url = URL.createObjectURL(blob)
          const anchor = document.createElement('a')
          anchor.href = url
          anchor.download = message.filename.split('/').pop() || 'download'
          anchor.click()
          setTimeout(() => URL.revokeObjectURL(url), 1000)
          pushNotification(`Fichier enregistré : ${anchor.download}`, 'info', 3000)
          break
        }
      }
    }

    return () => {
      cancelled = true
      worker.terminate()
      workerRef.current = null
      mediaRef.current.dispose()
      for (const pending of pendingRuns.current.values()) pending.reject(new Error('Moteur arrêté.'))
      pendingRuns.current.clear()
    }
  }, [pushNotification])

  const runOnWorker = useCallback(
    (nodes: GraphNode[], edges: GraphEdge[], previewNodeId: string | null, frames: Record<string, CapturedFrame>) => {
      const worker = workerRef.current
      if (!worker) return Promise.reject(new Error('Moteur non initialisé.'))

      const requestId = nextRequestId.current++
      const bitmaps = Object.values(frames).map((f) => f.bitmap)

      return new Promise<{ nodesData: Record<string, unknown>; frame: string | null; errors: Record<string, string> }>((resolve, reject) => {
        pendingRuns.current.set(requestId, { resolve, reject })
        const request: WorkerRequest = { type: 'run', requestId, nodes, edges, previewNodeId, frames }
        worker.postMessage(request, bitmaps)
      })
    },
    []
  )

  const execute = useCallback(async () => {
    if (!workerRef.current) return

    if (isRunning.current) {
      // A run is already in flight; remember that the graph moved under it.
      runQueued.current = true
      return
    }
    isRunning.current = true

    try {
      const { nodes, edges } = graphRef.current
      const resolvedNodes = resolveInputPaths(nodes)
      const frames = await mediaRef.current.captureFrames(nodes)
      const result = await runOnWorker(resolvedNodes, edges, previewNodeRef.current, frames)

      nodesDataRef.current = result.nodesData
      nodesDataStore._update(result.nodesData)
      setFrame(result.frame ? `data:image/jpeg;base64,${result.frame}` : null)

      for (const nodeId of captureRequests.current) {
        const preview = result.nodesData[`${nodeId}:main_preview`]
        if (typeof preview === 'string') onCapture?.(nodeId, preview)
      }
      captureRequests.current.clear()

      for (const [nodeId, message] of Object.entries(result.errors)) {
        const node = nodes.find((n) => n.id === nodeId)
        pushNotification(`${node?.type ?? nodeId} : ${message}`, 'error', 0)
      }
    } catch (error) {
      pushNotification(error instanceof Error ? error.message : String(error), 'error', 0)
    } finally {
      isRunning.current = false
      setComputingNodeId(null)
      if (runQueued.current) {
        runQueued.current = false
        void execute()
      }
    }
  }, [nodesDataStore, onCapture, pushNotification, runOnWorker])

  const scheduleRun = useCallback(() => {
    if (runTimer.current) clearTimeout(runTimer.current)
    runTimer.current = setTimeout(() => void execute(), RUN_DEBOUNCE_MS)
  }, [execute])

  const updateGraph = useCallback(
    (nodes: any[], edges: any[]) => {
      graphRef.current = {
        // Ink is pure canvas decoration with no ports — keep its stroke payload out.
        nodes: nodes.filter((n) => n.type !== 'canvas_ink').map((n) => ({ id: n.id, type: n.type, data: n.data })),
        edges,
      }
      if (isConnected) scheduleRun()
    },
    [isConnected, scheduleRun]
  )

  // The first graph usually arrives before OpenCV finishes loading; run it once ready.
  useEffect(() => {
    if (isConnected) scheduleRun()
  }, [isConnected, scheduleRun])

  const requestCapture = useCallback(
    (nodeId: string) => {
      const cached = nodesDataRef.current[`${nodeId}:main_preview`]
      if (typeof cached === 'string') {
        onCapture?.(nodeId, cached)
        return
      }
      captureRequests.current.add(nodeId)
      scheduleRun()
    },
    [onCapture, scheduleRun]
  )

  const setPreviewNode = useCallback(
    (nodeId: string | null) => {
      previewNodeRef.current = nodeId
      if (nodeId) setComputingNodeId(nodeId)
      scheduleRun()
    },
    [scheduleRun]
  )

  const requestSnapshotToNode = useCallback(() => {
    pushNotification('Snapshot indisponible dans la version en ligne.', 'warning')
  }, [pushNotification])

  const requestPyExport = useCallback(async (): Promise<string> => {
    // Generating the Python script means walking the desktop plugin sources,
    // which this build does not ship.
    throw new Error('L’export Python nécessite la version bureau de VNStudio.')
  }, [])

  const cancelNotification = useCallback((id: string) => dismissNotification(id), [dismissNotification])
  const retryInstall = useCallback((id: string) => dismissNotification(id), [dismissNotification])

  return {
    frame,
    nodesData: nodesDataRef.current,
    nodesDataStore,
    pluginSchemas,
    isConnected,
    updateGraph,
    requestCapture,
    requestSnapshotToNode,
    setPreviewNode,
    lastCommands: [] as any[],
    notifications,
    dismissNotification,
    cancelNotification,
    retryInstall,
    pushNotification,
    requestPyExport,
    computingNodeId,
  }
}
