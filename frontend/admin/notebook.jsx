import { useEffect, useRef, useState } from 'react'
import { Braces, Database, Download, FileUp, Play, Plus, RefreshCw, Square, Trash2 } from 'lucide-react'
import { ErrorNote, PanelTitle, Status } from './ui.jsx'
import { maxCells, maxNotebookBytes, newCell, parseNotebook, serializeNotebook, starterNotebook } from './notebook-format.js'
import './notebook.css'

const storageKey = 'sleepops-notebook-v1'

function readDraft() {
  try {
    const draft = JSON.parse(localStorage.getItem(storageKey))
    if (draft?.cells?.length && typeof draft.title === 'string') return draft
  } catch {
    // A browser can disable storage for embedded private Spaces.
  }
  return starterNotebook()
}

function useKernel() {
  const workerRef = useRef(null)
  const active = useRef(false)
  const execution = useRef(0)
  const [status, setStatus] = useState('idle')
  const [runningId, setRunningId] = useState(null)
  const [error, setError] = useState('')
  const [attachment, setAttachment] = useState(null)
  const [results, setResults] = useState({})

  useEffect(() => () => workerRef.current?.terminate(), [])

  function finish(next) {
    active.current = false
    setRunningId(null)
    setStatus(next)
  }

  function reset() {
    workerRef.current?.terminate()
    workerRef.current = null
    execution.current = 0
    setError('')
    setAttachment(null)
    finish('idle')
  }

  function send(job) {
    if (active.current) return
    active.current = true
    setError('')
    setStatus(workerRef.current ? (job.attach ? 'attaching' : 'running') : 'loading')
    try {
      if (!workerRef.current) {
        const worker = new Worker(new URL('./notebook.worker.js', import.meta.url), { type: 'module' })
        worker.onmessage = ({ data }) => {
          if (data.type === 'loading' || data.type === 'attaching') setStatus(data.type)
          if (data.type === 'running') {
            setRunningId(data.id)
            setStatus('running')
          }
          if (data.type === 'output') {
            const output = { ...data.output, sequence: ++execution.current }
            setResults((current) => ({ ...current, [data.id]: output }))
          }
          if (data.type === 'attached') setAttachment({ id: data.id, path: data.path })
          if (data.type === 'complete') finish('ready')
          if (data.type === 'error') {
            setError(data.message)
            finish('error')
          }
        }
        worker.onerror = (event) => {
          worker.terminate()
          workerRef.current = null
          setAttachment(null)
          setError(event.message || 'The Python worker could not start.')
          finish('error')
        }
        workerRef.current = worker
      }
      workerRef.current.postMessage(job)
    } catch (failure) {
      setError(failure.message)
      finish('error')
    }
  }

  return { status, runningId, attachment, error, results, send, reset,
    clearOutputs: () => setResults({}), forgetOutput: (id) => setResults((current) => {
      const next = { ...current }
      delete next[id]
      return next
    }) }
}

function Output({ result }) {
  if (!result) return null
  return <div className="cell-output" aria-live="polite">
    {(result.logs || []).map(({ kind, text }, index) => <pre key={index} className={kind === 'stderr' ? 'cell-stderr' : ''}>{text}</pre>)}
    {result.result && <pre className="cell-result">{result.result}</pre>}
    {result.error && <pre className="cell-stderr" role="alert">{result.error}</pre>}
    {(result.images || []).map((image, index) => <img key={index} src={`data:image/png;base64,${image}`}
      alt={`Python figure ${index + 1}`} />)}
  </div>
}

function NotebookCell({ cell, result, running, busy, onRun, onEdit, onRemove, removable }) {
  const code = cell.kind === 'code'
  return <article className={`notebook-cell panel ${running ? 'cell-active' : ''}`}>
    <div className="cell-toolbar">
      {code ? <button type="button" className="button secondary cell-run" title="Run cell (Shift+Enter)"
        disabled={busy || !cell.source.trim()} onClick={onRun}><Play size={14} /> Run cell</button>
        : <Status>{cell.kind === 'markdown' ? 'Markdown' : 'Raw text'}</Status>}
      {code && <span className="cell-number">{running ? '[*]' : `[${result?.sequence ?? ' '}]`}</span>}
      <button type="button" className="button notebook-remove" aria-label="Remove cell" title="Remove cell"
        disabled={busy || !removable} onClick={onRemove}><Trash2 size={16} /></button>
    </div>
    <textarea spellCheck="false" aria-label={code ? 'Python code' : 'Cell text'} value={cell.source}
      rows={Math.max(4, Math.min(16, cell.source.split('\n').length + 1))} onChange={(event) => onEdit(event.target.value)}
      onKeyDown={(event) => {
        if (code && event.shiftKey && event.key === 'Enter') {
          event.preventDefault()
          if (!busy && cell.source.trim()) onRun()
        }
      }} />
    <Output result={result} />
  </article>
}

export function Notebook({ versions, selected, onSelect, sample }) {
  const [draft, setDraft] = useState(readDraft)
  const [fileError, setFileError] = useState('')
  const picker = useRef(null)
  const kernel = useKernel()
  const { status, runningId, error, results, attachment } = kernel
  const busy = ['loading', 'attaching', 'running'].includes(status)
  const dataset = versions.find((version) => version.id === selected) || versions[0]
  const recording = dataset && sample?.version === dataset.id ? sample.data : null
  const context = JSON.stringify({ dataset: dataset || null, recording })
  const statusLabels = { idle: 'Kernel idle', ready: 'Kernel ready', loading: 'Starting Python', running: 'Running Python',
    attaching: 'Loading dataset', error: 'Kernel error' }

  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(draft))
    } catch {
      // Export is available when a browser blocks local storage.
    }
  }, [draft])

  function run(cells) {
    const code = cells.filter((cell) => cell.kind === 'code' && cell.source.trim())
    if (code.length) kernel.send({ cells: code, context })
  }

  async function open(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setFileError('')
    try {
      if (file.size > maxNotebookBytes) throw new Error('Choose a notebook up to 2 MiB.')
      const next = parseNotebook(await file.text(), file.name.replace(/\.ipynb$/i, ''))
      kernel.reset()
      kernel.clearOutputs()
      setDraft(next)
    } catch (failure) {
      setFileError(failure instanceof SyntaxError ? 'This file is not a valid Jupyter notebook.' : failure.message)
    }
  }

  function download() {
    const url = URL.createObjectURL(new Blob([serializeNotebook(draft, results)], { type: 'application/x-ipynb+json' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `${draft.title.trim() || 'SleepOps analysis'}.ipynb`
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  function edit(id, source) {
    setDraft((current) => ({ ...current, cells: current.cells.map((cell) => cell.id === id ? { ...cell, source } : cell) }))
  }

  function remove(id) {
    setDraft((current) => ({ ...current, cells: current.cells.filter((cell) => cell.id !== id) }))
    kernel.forgetOutput(id)
  }

  return <div className="notebook-layout">
    <div className="notebook-main">
      <section className="panel notebook-actions" aria-label="Notebook controls">
        <label htmlFor="notebook-name">Notebook name</label>
        <input id="notebook-name" value={draft.title} maxLength={80}
          onChange={(event) => setDraft((current) => ({ ...current, title: event.target.value }))} />
        <div className="notebook-buttons">
          <button type="button" className="button primary" disabled={busy || !draft.cells.some((cell) => cell.kind === 'code' && cell.source.trim())}
            onClick={() => run(draft.cells)}><Play size={15} /> Run all</button>
          <button type="button" className="button secondary" disabled={busy || draft.cells.length >= maxCells}
            onClick={() => setDraft((current) => ({ ...current, cells: [...current.cells, newCell()] }))}><Plus size={15} /> Add cell</button>
          <button type="button" className="button secondary" onClick={kernel.reset} disabled={status === 'idle'}>
            {busy ? <Square size={14} /> : <RefreshCw size={14} />}{busy ? 'Interrupt' : 'Reset kernel'}</button>
          <button type="button" className="button secondary" disabled={busy} onClick={() => picker.current.click()}><FileUp size={15} /> Open .ipynb</button>
          <button type="button" className="button secondary" onClick={download}><Download size={15} /> Export .ipynb</button>
          <input type="file" ref={picker} hidden accept=".ipynb,application/json" onChange={open} />
        </div>
        {fileError && <ErrorNote>{fileError}</ErrorNote>}
        {error && <ErrorNote>{error}</ErrorNote>}
      </section>
      {draft.cells.map((cell) => <NotebookCell key={cell.id} cell={cell} result={results[cell.id]}
        running={runningId === cell.id} busy={busy} removable={draft.cells.length > 1} onRun={() => run([cell])}
        onEdit={(source) => edit(cell.id, source)} onRemove={() => remove(cell.id)} />)}
    </div>
    <aside className="notebook-context">
      <section className="panel"><PanelTitle icon={Braces} title="Python kernel" />
        <p className="notebook-kernel-status" role="status"><Status tone={status === 'error' ? 'warning' : busy ? 'info' : 'positive'} dot>
          {statusLabels[status]}</Status></p>
        <p>Cells run in your browser. Use <kbd>Shift</kbd> + <kbd>Enter</kbd> to run a cell.</p>
        <a className="data-link" target="_blank" rel="noreferrer"
          href="https://colab.research.google.com/github/adam12bT/deepecg/blob/sleepops/Sleep_Stage_Awakening_Training_Colab.ipynb">Open two-stage training in Colab</a>
      </section>
      <section className="panel"><PanelTitle icon={Database} title="Notebook data" />
        <div className="field"><label htmlFor="notebook-dataset">Dataset version</label><select id="notebook-dataset"
          value={dataset?.id || ''} onChange={(event) => onSelect(event.target.value)}>
          {!versions.length && <option value="">Dataset unavailable</option>}
          {versions.map((version) => <option key={version.id} value={version.id}>{version.name}</option>)}
        </select></div>
        <p><code>dataset</code> contains this version’s split counts.</p>
        {recording ? <p><code>recording</code> contains {recording.values.length} epochs from {recording.id}.</p>
          : <p>Use <strong>Open in notebook</strong> in the Data tab to add a recording window.</p>}
        <button type="button" className="button secondary" disabled={!dataset || busy || attachment?.id === dataset.id}
          onClick={() => kernel.send({ attach: dataset.id })}><FileUp size={15} />
          {attachment?.id === dataset?.id && dataset ? 'Dataset attached' : 'Attach dataset file'}</button>
        {attachment?.id === dataset?.id && dataset && <p><code>dataset_path</code> points to the attached NPZ.</p>}
      </section>
    </aside>
  </div>
}
