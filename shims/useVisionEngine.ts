/**
 * Drop-in replacement for the desktop `useVisionEngine`.
 *
 * The desktop hook streams the graph to a Python process over a WebSocket and
 * receives rendered frames back. This one keeps the identical return shape but
 * evaluates the graph in-page with OpenCV.js, so the app runs from a static
 * host with no backend. Every consumer in src/ is untouched.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createNodesDataStore } from '../src/context/NodesDataContext'
import { loadOpenCv, offOpenCvProgress } from '../web-engine/opencv'
import { GraphExecutor, type GraphEdge, type GraphNode } from '../web-engine/executor'
import { SCHEMAS } from '../web-engine/registry'

export type EngineNotification = {
  id: string
  message: string
  progress: number | null
  level: 'info' | 'warning' | 'error'
}

/** Coalesces the bursts of graph updates React emits while a slider is dragged. */
const RUN_DEBOUNCE_MS = 60

export function useVisionEngine(onCapture?: (nodeId: string, base64: string) => void) {
  const [frame, setFrame] = useState<string | null>(null)
  const nodesDataStore = useMemo(() => createNodesDataStore(), [])
  const nodesDataRef = useRef<Record<string, any>>({})
  const [pluginSchemas] = useState<any[]>(SCHEMAS)
  const [isConnected, setIsConnected] = useState(false)
  const [notifications, setNotifications] = useState<EngineNotification[]>([])
  const [computingNodeId, setComputingNodeId] = useState<string | null>(null)

  const executorRef = useRef<GraphExecutor | null>(null)
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

    const showProgress = ({ progress, message }: { progress: number | null; message: string }) => {
      if (cancelled) return
      setNotifications((prev) => {
        const entry: EngineNotification = { id: LOADING_ID, message, progress, level: 'info' }
        const index = prev.findIndex((n) => n.id === LOADING_ID)
        return index >= 0 ? prev.map((n, i) => (i === index ? entry : n)) : [...prev, entry]
      })
    }

    showProgress({ progress: 0, message: 'Téléchargement d’OpenCV.js…' })

    loadOpenCv(showProgress)
      .then((cv) => {
        if (cancelled) return
        executorRef.current = new GraphExecutor(cv)
        setIsConnected(true)
        setNotifications((prev) => prev.filter((n) => n.id !== LOADING_ID))
        pushNotification('Moteur navigateur prêt ✓', 'info', 2500)
      })
      .catch((error: unknown) => {
        if (cancelled) return
        const message = error instanceof Error ? error.message : String(error)
        setNotifications((prev) =>
          prev.map((n) => (n.id === LOADING_ID ? { ...n, message, progress: null, level: 'error' } : n))
        )
      })

    return () => {
      cancelled = true
      offOpenCvProgress(showProgress)
      executorRef.current?.dispose()
      executorRef.current = null
    }
  }, [pushNotification])

  const execute = useCallback(async () => {
    const executor = executorRef.current
    if (!executor) return

    if (isRunning.current) {
      // A run is already in flight; remember that the graph moved under it.
      runQueued.current = true
      return
    }
    isRunning.current = true

    try {
      const { nodes, edges } = graphRef.current
      const result = await executor.run(nodes, edges, previewNodeRef.current)

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
    } finally {
      isRunning.current = false
      setComputingNodeId(null)
      if (runQueued.current) {
        runQueued.current = false
        void execute()
      }
    }
  }, [nodesDataStore, onCapture, pushNotification])

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
