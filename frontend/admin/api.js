import { useEffect, useState } from 'react'

export async function request(path, options) {
  const response = await fetch(`/api/admin${path}`, { cache: 'no-store', ...options })
  const body = await response.json().catch(() => null)
  if (!response.ok || !body) {
    throw new Error(typeof body?.detail === 'string' ? body.detail : 'The admin service could not complete this request.')
  }
  return body
}

export function useResource(path, revision = 0, retain = false) {
  const [result, setResult] = useState(null)
  useEffect(() => {
    if (!path) return
    const controller = new AbortController()
    request(path, { signal: controller.signal }).then(
      (data) => setResult({ path, revision, data }),
      (error) => { if (!controller.signal.aborted) setResult({ path, revision, error: error.message }) },
    )
    return () => controller.abort()
  }, [path, revision])
  return result?.path === path && (result?.revision === revision || (retain && result?.data)) ? result : null
}
