import { useEffect, useState } from 'react'

declare global {
  interface Window {
    cv: any
    Module?: any
  }
}

const OPENCV_URL = 'https://docs.opencv.org/4.9.0/opencv.js'

let loadPromise: Promise<any> | null = null

function loadOpenCv(): Promise<any> {
  if (loadPromise) return loadPromise
  loadPromise = new Promise((resolve, reject) => {
    if (window.cv && window.cv.Mat) {
      resolve(window.cv)
      return
    }
    const script = document.createElement('script')
    script.src = OPENCV_URL
    script.async = true
    script.onload = () => {
      const cv = window.cv
      if (!cv) {
        reject(new Error('opencv.js loaded but window.cv missing'))
        return
      }
      if (cv.getBuildInformation) {
        resolve(cv)
      } else {
        cv.onRuntimeInitialized = () => resolve(cv)
      }
    }
    script.onerror = () => reject(new Error('failed to load opencv.js'))
    document.head.appendChild(script)
  })
  return loadPromise
}

export function useOpenCv() {
  const [cv, setCv] = useState<any>(window.cv?.Mat ? window.cv : null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    loadOpenCv()
      .then((loaded) => {
        if (!cancelled) setCv(loaded)
      })
      .catch((e) => {
        if (!cancelled) setError(String(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { cv, ready: !!cv, error }
}
