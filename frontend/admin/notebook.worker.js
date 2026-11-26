const runtimeRoot = new URL(`${import.meta.env.BASE_URL}python/`, self.location.origin).href
const textLimit = 24000
let kernel
let loadedContext
let attachedDataset
const archivePath = '/data/sleep_edf_wake_data.npz'

async function initialize() {
  const { loadPyodide } = await import(/* @vite-ignore */ `${runtimeRoot}pyodide.mjs`)
  return loadPyodide({
    indexURL: runtimeRoot,
    packageBaseUrl: runtimeRoot,
    env: { MPLBACKEND: 'Agg' },
  })
}

const snapshots = `
def _sleepops_render_figures():
    import sys
    from io import BytesIO
    from base64 import b64encode
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is None:
        return []
    plots = []
    for label in pyplot.get_fignums()[-3:]:
        figure = pyplot.figure(label)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=100)
        payload = buffer.getvalue()
        if len(payload) <= 650_000:
            plots.append(b64encode(payload).decode("ascii"))
        pyplot.close(figure)
    return plots
_sleepops_render_figures()
`

async function runCell(python, cell) {
  const logs = []
  let size = 0
  function capture(kind, value) {
    if (size >= textLimit) return
    const text = `${value}\n`.slice(0, textLimit - size)
    const last = logs.at(-1)
    if (last?.kind === kind) last.text += text
    else logs.push({ kind, text })
    size += text.length
    if (size === textLimit) logs.push({ kind, text: '\nOutput truncated.\n' })
  }
  try {
    python.setStdout({ batched: () => {} })
    python.setStderr({ batched: (text) => capture('stderr', text) })
    await python.loadPackagesFromImports(cell.source)
    python.setStdout({ batched: (text) => capture('stdout', text) })
    const value = await python.runPythonAsync(cell.source)
    let result = ''
    try {
      if (value !== undefined && value !== null) result = String(value).slice(0, textLimit)
    } finally {
      value?.destroy?.()
    }
    const plots = python.runPython(snapshots)
    try {
      return { logs, result, images: plots.toJs() }
    } finally {
      plots.destroy()
    }
  } catch (error) {
    return { logs, error: String(error.message || error).slice(0, textLimit) }
  }
}

self.onmessage = async ({ data }) => {
  const { cells, context, attach } = data
  try {
    if (!kernel) {
      self.postMessage({ type: 'loading' })
      kernel = await initialize()
    }
    if (attach) {
      self.postMessage({ type: 'attaching' })
      const response = await fetch(`/api/admin/datasets/${encodeURIComponent(attach)}/archive`, { cache: 'no-store' })
      if (!response.ok) throw new Error('The selected dataset could not be loaded into Python.')
      kernel.FS.mkdirTree('/data')
      kernel.FS.writeFile(archivePath, new Uint8Array(await response.arrayBuffer()))
      attachedDataset = attach
      loadedContext = undefined
      self.postMessage({ type: 'attached', id: attach, path: archivePath })
      self.postMessage({ type: 'complete' })
      return
    }
    if (context !== loadedContext) {
      kernel.globals.set('_sleepops_context', context)
      kernel.globals.set('dataset_path', JSON.parse(context).dataset?.id === attachedDataset ? archivePath : null)
      kernel.runPython(`
import json as _sleepops_json
_sleepops_data = _sleepops_json.loads(_sleepops_context)
dataset = _sleepops_data["dataset"]
recording = _sleepops_data["recording"]
del _sleepops_context, _sleepops_data
`)
      loadedContext = context
    }
    for (const cell of cells) {
      self.postMessage({ type: 'running', id: cell.id })
      const output = await runCell(kernel, cell)
      self.postMessage({ type: 'output', id: cell.id, output })
      if (output.error) break
    }
    self.postMessage({ type: 'complete' })
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error.message || error).slice(0, 1600) })
  }
}
