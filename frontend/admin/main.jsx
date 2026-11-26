import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, ChartNoAxesCombined, Check, ChevronDown, CircleAlert, CircleCheck, Cpu, Database,
  FlaskConical, GitCompare, Layers3, LayoutDashboard, LockKeyhole, MoonStar, NotebookPen, Play, RefreshCw, ScanLine,
  Rocket, ShieldCheck, Waves, X,
} from 'lucide-react'
import '@fontsource-variable/inter'
import '@fontsource-variable/space-grotesk'
import { request, useResource } from './api.js'
import { DatasetPage } from './data.jsx'
import { Notebook } from './notebook.jsx'
import { ErrorNote, PanelTitle, Status } from './ui.jsx'
import './styles.css'

const pages = {
  overview: ['Overview', LayoutDashboard],
  models: ['Model registry', Layers3],
  behavior: ['Model behavior', ScanLine],
  data: ['Data', Database],
  notebook: ['Notebook', NotebookPen],
  compare: ['Compare models', GitCompare],
  training: ['Training', FlaskConical],
}

const idle = { phase: 'idle', running: false }
const phases = {
  idle: ['Ready to train', 'Choose your settings and start a run.'],
  restoring: ['Restoring data', 'Verifying the DVC-pinned EEG archive.'],
  fitting: ['Training in progress', 'Selecting the checkpoint with the lowest validation loss.'],
  saving: ['Saving candidate', 'Writing and verifying the checkpoint in private storage.'],
  complete: ['Run complete', 'The new candidate is available in the model registry.'],
  failed: ['Training failed', 'Open the Space logs for details.'],
}

const percent = (value) => `${(value * 100).toFixed(2)}%`
const short = (id) => id.slice(0, 12)
const date = (value) => new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(value))

function Summary({ models }) {
  const active = models.find((model) => model.deployed)
  const candidates = models.filter((model) => !model.deployed)
  const cards = [
    ['Deployed release', active ? short(active.id) : '—', active ? `Selected at epoch ${active.epoch}` : 'Release unavailable', Layers3, 'green'],
    ['Saved candidates', String(candidates.length), `${candidates.filter((model) => model.gatesPassed).length} passed validation gates`, Database, 'blue'],
    ['Sleep-stage macro F1', active?.stageMacroF1 != null ? percent(active.stageMacroF1) : '—', 'Deployed · validation', MoonStar, 'teal'],
    ['Balanced accuracy', active ? percent(active.accuracy) : '—', 'Deployed · validation', Activity, 'violet'],
    ['Forecast log loss', active ? active.logLoss.toFixed(4) : '—', 'Deployed · validation', ChartNoAxesCombined, 'amber'],
  ]
  return <section className="metric-grid" aria-label="Model summary">
    {cards.map(([label, value, hint, Icon, color]) => <div className="metric-card" key={label}>
      <div className="metric-label"><span className={`metric-icon ${color}`}><Icon size={17} /></span>{label}</div>
      <strong className={label === 'Deployed release' ? 'release-number' : ''}>{value}</strong>
      <span className="metric-hint">{hint}</span>
    </div>)}
  </section>
}

function ModelSelect({ id, label, value, models, onChange }) {
  return <div className="field"><label htmlFor={id}>{label}</label><span className="select-wrap">
    <select id={id} value={value || ''} onChange={(event) => onChange(event.target.value)}>
      {!models.length && <option value="">No releases available</option>}
      {models.map((model) => <option key={model.key} value={model.key}>{model.label}</option>)}
    </select><ChevronDown size={16} />
  </span></div>
}

function ComparePanel({ models, revision }) {
  const [selectedA, setA] = useState('')
  const [selectedB, setB] = useState('')
  const [split, setSplit] = useState('validation')
  const contains = (key) => models.some((model) => model.key === key)
  const a = contains(selectedA) ? selectedA : models[0]?.key
  const b = contains(selectedB) ? selectedB : (models.find((model) => model.key !== a && model.gatesPassed)?.key
    ?? models.find((model) => model.key !== a)?.key ?? a)
  const path = a && b ? `/comparison?${new URLSearchParams({ a, b, split })}` : null
  const result = useResource(path, revision)
  const comparison = result?.data

  return <section className="panel compare-panel">
    <PanelTitle icon={GitCompare} title="Compare models">
      <div className="segment" aria-label="Evaluation split">
        <button type="button" aria-pressed={split === 'validation'} onClick={() => setSplit('validation')}>Validation</button>
        <button type="button" aria-pressed={split === 'test'} onClick={() => setSplit('test')}>Held-out test</button>
      </div>
    </PanelTitle>
    <div className="pair-fields">
      <ModelSelect id="model-a" label="Model A" value={a} models={models} onChange={setA} />
      <ModelSelect id="model-b" label="Model B" value={b} models={models} onChange={setB} />
    </div>
    {result?.error && <ErrorNote>{result.error}</ErrorNote>}
    {!result && <p className="notice">{path ? 'Loading comparison…' : 'Add a model to see evaluation results.'}</p>}
    {comparison && <>
      <div className="table-scroll"><table className="comparison-table"><thead><tr>
        <th scope="col">Metric</th><th scope="col"><span className="model-key a">A</span> Model A</th>
        <th scope="col"><span className="model-key b">B</span> Model B</th><th scope="col">Δ B − A</th>
      </tr></thead><tbody>{comparison.rows.map(([name, first, second, delta, better]) => <tr key={name}>
        <th scope="row">{name}</th>
        <td className={better === 'A' ? 'better' : ''}>{first}{better === 'A' && <Check size={14} aria-label="Better result" />}</td>
        <td className={better === 'B' ? 'better' : ''}>{second}{better === 'B' && <Check size={14} aria-label="Better result" />}</td>
        <td className="delta">{delta}</td>
      </tr>)}</tbody></table></div>
      <p className="comparison-note">{comparison.comparable ? <CircleCheck size={16} /> : <CircleAlert size={16} />}{comparison.message}</p>
    </>}
  </section>
}

function clock(seconds) {
  const value = Math.max(0, Number(seconds) || 0)
  return [Math.floor(value / 3600), Math.floor((value % 3600) / 60), value % 60]
    .map((part) => String(part).padStart(2, '0')).join(':')
}

function ModelBehavior({ models, revision }) {
  const stageModels = models.filter((model) => model.stageMacroF1 != null)
  const [selectedKey, setSelectedKey] = useState('')
  const selectedModel = stageModels.find((model) => model.key === selectedKey) ?? stageModels[0]
  const key = selectedModel?.key ?? ''
  const optionsResult = useResource(key ? `/models/${encodeURIComponent(key)}/behavior-options` : null, revision)
  const options = optionsResult?.data
  const recordings = options?.recordings ?? []
  const [recordId, setRecordId] = useState('')
  const [windowIndex, setWindowIndex] = useState(0)
  const recording = recordings.find((item) => item.id === recordId) ?? recordings[0]
  const ends = recording?.windowEndsSec ?? []
  const index = Math.min(windowIndex, Math.max(ends.length - 1, 0))
  const end = ends[index]
  const params = end == null ? null : new URLSearchParams({ record: recording.id, end_sec: String(end) })
  const behaviorResult = useResource(params ? `/models/${encodeURIComponent(key)}/behavior?${params}` : null, revision)
  const behavior = behaviorResult?.data

  useEffect(() => {
    if (!recordings.some((item) => item.id === recordId)) {
      setRecordId(recordings[0]?.id ?? '')
      setWindowIndex(0)
    }
  }, [recordings, recordId])

  useEffect(() => {
    if (windowIndex >= ends.length) setWindowIndex(Math.max(ends.length - 1, 0))
  }, [ends.length, windowIndex])

  return <section className="panel behavior-panel">
    <PanelTitle icon={ScanLine} title="Inspect model behavior">
      {selectedModel && <span className={`badge ${selectedModel.gatesPassed ? 'positive' : 'warning'}`}>
        {selectedModel.gatesPassed ? 'Passed validation gates' : 'Candidate · review gates'}
      </span>}
    </PanelTitle>
    {!stageModels.length && <p className="notice">No five-stage model is in the registry yet. Add a stage-aware candidate to inspect its predictions here.</p>}
    {stageModels.length > 0 && <>
      <div className="behavior-controls">
        <ModelSelect id="behavior-model" label="Stage-aware model" value={key} models={stageModels} onChange={setSelectedKey} />
        {recordings.length > 0 && <label className="field" htmlFor="behavior-recording">Held-out recording
          <span className="select-wrap"><select id="behavior-recording" value={recording?.id ?? ''}
            onChange={(event) => { setRecordId(event.target.value); setWindowIndex(0) }}>
            {recordings.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
          </select><ChevronDown size={16} /></span>
        </label>}
      </div>
      {optionsResult?.error && <ErrorNote>{optionsResult.error}</ErrorNote>}
      {!optionsResult && <p className="notice">Loading held-out recording windows…</p>}
      {recording && ends.length > 0 && <div className="window-slider">
        <label htmlFor="behavior-window">10 completed EEG epochs ending at <strong>{clock(end)}</strong></label>
        <input id="behavior-window" type="range" min="0" max={ends.length - 1} value={index}
          aria-valuetext={`${clock(end)} from recording start`}
          onChange={(event) => setWindowIndex(Number(event.target.value))} />
        <div><span>{clock(ends[0])}</span><span>{clock(ends.at(-1))}</span></div>
        <p className="notice">Held-out test recording · input covers {clock(end - 300)}–{clock(end)}. The model receives only these past epochs.</p>
      </div>}
      {behaviorResult?.error && <ErrorNote>{behaviorResult.error}</ErrorNote>}
      {!behaviorResult && end != null && <p className="notice">Running this checkpoint on the selected window…</p>}
      {behavior && <>
        <div className="behavior-summary">
          <div><small>Current sleep state</small><strong>{behavior.state === 'asleep' ? 'Asleep' : 'Awake'}</strong>
            <span>{percent(behavior.asleepProbability)} probability of sleep</span></div>
          <div><small>Current stage</small><strong>{behavior.predictedStages.at(-1) ?? 'Not available'}</strong>
            <span>{behavior.stageProbabilities.length ? percent(Math.max(...behavior.stageProbabilities.at(-1))) + ' classifier confidence' : 'This release has no stage head'}</span></div>
          <div><small>Selected recording</small><strong>{behavior.recordingId}</strong>
            <span>Prediction time {clock(behavior.endSec)}</span></div>
        </div>
        {behavior.stageProbabilities.length > 0 && <>
          <h3 className="behavior-subtitle">Sleep-stage timeline <span>Predicted vs scored</span></h3>
          <div className="behavior-legend">{behavior.stageNames.map((stage, stageIndex) => <span key={stage}>
            <i className={`stage-dot stage-${stageIndex}`} />{stage}</span>)}</div>
          <div className="behavior-epochs">
            {behavior.predictedStages.map((stage, epochIndex) => {
              const stageIndex = behavior.stageNames.indexOf(stage)
              const confidence = behavior.stageProbabilities[epochIndex][stageIndex]
              return <div className={`behavior-epoch stage-${stageIndex}`} key={epochIndex}>
                <small>{epochIndex === 9 ? 'Now' : `−${(9 - epochIndex) * 30}s`}</small>
                <strong>{stage}</strong><span>{percent(confidence)}</span>
                <em>Scored: {behavior.observedStages[epochIndex]}</em>
              </div>
            })}
          </div>
        </>}
        {behavior.forecast ? <>
          <h3 className="behavior-subtitle">Awakening forecast <span>13 class probabilities</span></h3>
          <div className="behavior-forecast">
            {behavior.forecast.bins.map((bin) => <div className="behavior-bin" key={bin.interval}>
              <span>{bin.interval}</span><div><i style={{ width: `${bin.probability * 100}%` }} /></div>
              <strong>{percent(bin.probability)}</strong>
            </div>)}
          </div>
        </> : <p className="notice behavior-sleep-note">The forecast head is shown when this window predicts sleep.</p>}
      </>}
    </>}
  </section>
}

function TrainingPanel({ status, onStart, versions, dataset, onDataset }) {
  const [epochs, setEpochs] = useState('20')
  const [evaluateTest, setEvaluateTest] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [title, context] = phases[status.phase] || phases.idle
  const locked = Boolean(status.running || submitting)

  async function submit(event) {
    event.preventDefault()
    setError('')
    setSubmitting(true)
    try {
      await onStart(Number(epochs), evaluateTest)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setSubmitting(false)
    }
  }

  return <section className="panel training-panel">
    <PanelTitle icon={FlaskConical} title="Train a model"><Status><Cpu size={14} /> CPU Basic</Status></PanelTitle>
    <div className={`run-state ${status.phase}`} role="status">
      <span className="run-icon">{status.running ? <RefreshCw className="spin" size={17} /> : <Activity size={17} />}</span>
      <div><strong>{title}</strong><p>{context}</p></div>
    </div>
    <form onSubmit={submit}>
      <label className="field" htmlFor="training-dataset">Training dataset</label>
      <div className="select-wrap training-select"><select id="training-dataset" value={dataset} disabled={locked}
        onChange={(event) => onDataset(event.target.value)}>
        {(versions.length ? versions : [{ id: 'base', name: 'Original Sleep-EDF' }]).map((version) =>
          <option key={version.id} value={version.id}>{version.name} · {version.splits?.train?.stage_counts ? 'five sleep stages' : 'sleep/wake labels'}</option>)}
      </select><ChevronDown size={16} /></div>
      <label className="epoch-field field" htmlFor="epochs"><span>Maximum epochs</span>
        <input type="number" id="epochs" required min="1" max="20" step="1" value={epochs}
          onChange={(event) => setEpochs(event.target.value)} disabled={locked} />
      </label>
      <label className="test-option"><span>Evaluate on held-out subjects</span>
        <input type="checkbox" checked={evaluateTest} onChange={(event) => setEvaluateTest(event.target.checked)} disabled={locked} />
      </label>
      <button className="button primary run-button" disabled={locked || !epochs}><Play size={16} />
        {locked ? 'Run in progress' : 'Start training'}
      </button>
    </form>
    {error && <ErrorNote>{error}</ErrorNote>}
    {status.result && <a className="run-result" href={status.result.url} target="_blank" rel="noreferrer">
      <CircleCheck size={17} /> <span>Epoch {status.result.epoch} selected · loss {status.result.loss.toFixed(5)}</span>
    </a>}
  </section>
}

const deploymentWorkflow = 'https://github.com/adam12bT/deepecg/actions/workflows/promote.yml'

function Deployment({ revision, onRefresh }) {
  const [poll, setPoll] = useState(0)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [queued, setQueued] = useState(null)
  const result = useResource('/deployment', revision + poll, true)
  const active = result?.data?.active
  const candidate = result?.data?.candidate
  const current = result?.data?.request
  const job = queued && current?.id !== queued.id ? queued : current
  const busy = sending || ['queued', 'running'].includes(job?.phase)
  const completed = useRef(null)

  useEffect(() => {
    const timer = setInterval(() => setPoll((value) => value + 1), 10000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (queued && current?.id === queued.id) setQueued(null)
  }, [queued, current?.id])

  useEffect(() => {
    if (job?.phase === 'succeeded' && completed.current !== job.id) {
      completed.current = job.id
      onRefresh()
    }
  }, [job?.phase, job?.id, onRefresh])

  async function deploy() {
    setSending(true)
    setError('')
    try {
      const saved = await request('/deployment', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate: candidate.id }) })
      setQueued(saved)
      setPoll((value) => value + 1)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setSending(false)
    }
  }

  const jobLabel = { queued: 'Queued', running: 'Verifying and deploying', succeeded: 'Live', failed: 'Deployment failed' }
  return <section className="panel deployment-panel">
    <PanelTitle icon={Rocket} title="Production deployment"
      description="Deploy the candidate with the lowest validation loss among models that pass the release gates.">
      <button className="button primary" type="button" disabled={!candidate || busy || result?.error || job?.phase === 'succeeded' && job.candidate === candidate?.id}
        onClick={deploy}><Rocket size={16} />{sending ? 'Queueing…' : 'Deploy best model'}</button>
    </PanelTitle>
    <div className="deployment-models">
      <div><small>Live release</small><strong>{active ? short(active.id) : 'Unavailable'}</strong>
        {active && <span>Validation loss {active.loss.toFixed(5)}</span>}</div>
      <div><small>Best eligible candidate</small><strong>{candidate ? short(candidate.id) : 'None available'}</strong>
        {candidate && <span>Validation loss {candidate.loss.toFixed(5)}</span>}</div>
    </div>
    {(error || result?.error) && <ErrorNote>{error || result.error}</ErrorNote>}
    <div className="deployment-status" aria-live="polite">
      {job && <Status dot tone={job.phase === 'succeeded' ? 'positive' : 'warning'}>{jobLabel[job.phase]}</Status>}
      {job?.phase === 'queued' && <span>GitHub checks the deployment queue about every 10 minutes.</span>}
      {job?.phase === 'failed' && <span>Open the workflow logs for details.</span>}
      <a href={job?.url || deploymentWorkflow} target="_blank" rel="noreferrer">GitHub Actions</a>
    </div>
  </section>
}

function Registry({ models, onInspect, onRefresh }) {
  return <section className="panel registry-panel">
    <PanelTitle icon={Layers3} title="Model registry" description="Deployed release and saved training candidates">
      <button type="button" className="button secondary" onClick={onRefresh}><RefreshCw size={15} /> Refresh models</button>
    </PanelTitle>
    {!models.length && <p className="notice">No model reports are available yet.</p>}
    {!!models.length && <div className="table-scroll"><table className="registry-table"><thead><tr>
      <th scope="col">Model version</th><th scope="col">Validation gates</th><th scope="col">Epoch</th>
      <th scope="col">Stage macro F1</th><th scope="col">Balanced accuracy</th><th scope="col">Forecast log loss</th><th scope="col" aria-label="Actions" />
    </tr></thead><tbody>{models.map((model) => <tr key={model.key}>
      <th scope="row"><div className="registry-model"><span className="registry-icon"><Layers3 size={17} /></span>
        <span><strong>{short(model.id)}</strong><small>{model.deployed ? 'Deployed release' : date(model.createdAt)}</small></span>
      </div></th>
      <td><Status tone={model.gatesPassed ? 'positive' : 'warning'} dot>{model.gatesPassed ? 'Passed' : 'Review'}</Status></td>
      <td>{model.epoch}</td><td>{model.stageMacroF1 == null ? '—' : percent(model.stageMacroF1)}</td>
      <td>{percent(model.accuracy)}</td><td>{model.logLoss.toFixed(4)}</td>
      <td><button type="button" className="button inspect-button" onClick={() => onInspect(model.key)}
        aria-label={`Inspect model ${short(model.id)}`}>Inspect</button></td>
    </tr>)}</tbody></table></div>}
  </section>
}

function ModelDialog({ modelKey, onClose }) {
  const ref = useRef(null)
  const result = useResource(`/models/${encodeURIComponent(modelKey)}`)
  const model = result?.data
  const [check, setCheck] = useState(null)

  useEffect(() => {
    const dialog = ref.current
    dialog.showModal()
    return () => dialog.close()
  }, [])

  async function verify() {
    setCheck({ pending: true })
    try {
      const data = await request(`/models/${encodeURIComponent(modelKey)}/check`, { method: 'POST' })
      setCheck(data)
    } catch (failure) {
      setCheck({ error: failure.message })
    }
  }

  return <dialog ref={ref} className="model-dialog" aria-labelledby="model-dialog-title"
    onCancel={onClose} onClick={(event) => { if (event.target === ref.current) onClose() }}>
    <div className="dialog-content">
      <div className="dialog-heading"><div><span className="eyebrow">Model details</span>
        <h2 id="model-dialog-title">{model?.label || 'Selected model'}</h2></div>
      <button type="button" className="button icon-button" aria-label="Close model details" onClick={onClose}><X size={20} /></button></div>
      {!result && <p className="notice">Loading model report…</p>}
      {result?.error && <ErrorNote>{result.error}</ErrorNote>}
      {model && <>
        <div className="model-meta"><span>Epoch <strong>{model.epoch}</strong></span>
          <span>Dataset <code>{short(model.dataset)}</code></span>
          <Status tone={model.gatesPassed ? 'positive' : 'warning'} dot>{model.gatesPassed ? 'Gates passed' : 'Review gates'}</Status>
        </div>
        {!model.gatesPassed && <p className="notice warning-text">Review {Object.entries(model.checks)
          .filter(([, passed]) => !passed).map(([name]) => name.replaceAll('_', ' ')).join(', ')}.</p>}
        <div className="table-scroll"><table className="details-table"><thead><tr>
          <th scope="col">Metric</th><th scope="col">Validation</th><th scope="col">Held-out test</th>
        </tr></thead><tbody>{model.metrics.map(([label, validation, test]) => <tr key={label}>
          <th scope="row">{label}</th><td>{validation}</td><td>{test}</td>
        </tr>)}</tbody></table></div>
        <div className="dialog-actions">
          <button type="button" className="button primary" onClick={verify} disabled={check?.pending}>
            <ScanLine size={16} />{check?.pending ? 'Checking…' : 'Check checkpoint'}
          </button>
          <a className="button secondary" href={model.url} target="_blank" rel="noreferrer">Open artifacts</a>
        </div>
        <div aria-live="polite">{check?.error && <ErrorNote>{check.error}</ErrorNote>}
          {check?.message && <p className="notice success"><ShieldCheck size={18} />{check.message}</p>}
        </div>
      </>}
    </div>
  </dialog>
}

function Sidebar({ page, onNavigate, catalog }) {
  return <aside className="sidebar">
    <a className="brand" href="#overview" aria-label="SleepOps overview">
      <span className="brand-icon"><Waves size={24} /></span><span><strong>SleepOps</strong><small>Admin console</small></span>
    </a>
    <nav className="navigation" aria-label="Admin navigation">
      {Object.entries(pages).map(([key, [label, Icon]]) => <button key={key} type="button"
        className={page === key ? 'current' : ''} aria-current={page === key ? 'page' : undefined}
        onClick={() => onNavigate(key)}><Icon size={18} />{label}</button>)}
    </nav>
    <div className="sidebar-footer">
      {catalog?.prediction && <a href={catalog.prediction} target="_blank" rel="noreferrer"><Activity size={16} /> Prediction demo</a>}
      {catalog?.bucket && <a href={catalog.bucket} target="_blank" rel="noreferrer"><Database size={16} /> Candidate artifacts</a>}
      {catalog?.dvc && <a href={catalog.dvc} target="_blank" rel="noreferrer"><Layers3 size={16} /> DVC model &amp; data</a>}
      <span className="private"><LockKeyhole size={14} /> Private workspace</span>
    </div>
  </aside>
}

function currentPage() {
  const page = window.location.hash.slice(1)
  return Object.hasOwn(pages, page) ? page : 'overview'
}

function App() {
  const [page, setPage] = useState(currentPage)
  const [modelKey, setModelKey] = useState(null)
  const [revision, setRevision] = useState(0)
  const [dataRevision, setDataRevision] = useState(0)
  const [viewDataset, setViewDataset] = useState('base')
  const [notebookDataset, setNotebookDataset] = useState('base')
  const [notebookSample, setNotebookSample] = useState(null)
  const [trainingDataset, setTrainingDataset] = useState('base')
  const [worker, setWorker] = useState(null)
  const [workerError, setWorkerError] = useState('')
  const catalog = useResource('/catalog', revision)
  const datasets = useResource('/datasets', dataRevision, true)
  const models = catalog?.data?.models || []
  const completed = useRef(null)
  const requestRevision = useRef(0)
  const refresh = () => setRevision((current) => current + 1)

  useEffect(() => {
    const onHash = () => setPage(currentPage())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    let timer
    let stopped = false
    const controller = new AbortController()
    async function poll() {
      const revisionAtStart = requestRevision.current
      try {
        const data = await request('/training', { signal: controller.signal })
        if (stopped || revisionAtStart !== requestRevision.current) return
        setWorker(data)
        setWorkerError('')
        if (data.result?.id && completed.current !== data.result.id) {
          completed.current = data.result.id
          setRevision((current) => current + 1)
        }
      } catch (error) {
        if (!stopped && revisionAtStart === requestRevision.current) setWorkerError(error.message)
      } finally {
        if (!stopped) timer = setTimeout(poll, 3500)
      }
    }
    poll()
    return () => { stopped = true; clearTimeout(timer); controller.abort() }
  }, [])

  async function start(epochs, evaluateTest) {
    requestRevision.current += 1
    const data = await request('/training', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs, evaluateTest, dataset: trainingDataset }) })
    requestRevision.current += 1
    setWorker(data)
    setWorkerError('')
  }

  function navigate(next) {
    setPage(next)
    window.location.hash = next
    window.scrollTo({ top: 0 })
  }

  const overview = page === 'overview'
  return <div className="app-shell">
    <Sidebar page={page} onNavigate={navigate} catalog={catalog?.data} />
    <main>
      <header className="topbar"><span>{pages[page][0]}</span><Status
        tone={workerError ? 'warning' : worker?.running ? 'info' : worker ? 'positive' : 'muted'} dot>
        {workerError ? 'Worker unavailable' : worker?.running ? 'Training active' : worker ? 'Worker ready' : 'Connecting'}</Status></header>
      <div className="workspace">
        <div className="page-heading"><div><h1>{overview ? 'Model operations' : pages[page][0]}</h1>
          <p>{overview ? 'Monitor your SleepOps release and train new candidates.'
              : { models: 'Inspect deployed and candidate checkpoints.', behavior: 'Replay held-out windows through a saved model and review its stage and awakening probabilities.', data: 'Explore EEG recordings and manage versioned training data.',
              notebook: 'Write Python cells and work with your EEG data.',
              compare: 'Evaluate two model versions side by side.',
              training: 'Choose a versioned EEG dataset and train a new review candidate.' }[page]}</p></div>
        {page !== 'training' && page !== 'notebook' && <button type="button" className="button primary" onClick={() => navigate('training')}>
          <Play size={16} /> Training controls</button>}</div>

        {catalog?.error && <ErrorNote>{catalog.error}</ErrorNote>}
        {!catalog && <p className="notice" role="status">Loading model registry…</p>}
        {workerError && <ErrorNote>{workerError}</ErrorNote>}
        {page === 'data' && <DatasetPage resource={datasets} selected={viewDataset} onSelect={setViewDataset}
          onSaved={(version) => { setViewDataset(version.id); setTrainingDataset(version.id); setDataRevision((current) => current + 1) }}
          onNotebook={(sample) => { setNotebookDataset(sample.version); setNotebookSample(sample); navigate('notebook') }}
          onTrain={(id) => { setTrainingDataset(id); navigate('training') }} />}
        {page === 'behavior' && <ModelBehavior models={models} revision={revision} />}
        <div hidden={page !== 'notebook'}><Notebook versions={datasets?.data?.versions || []}
          selected={notebookDataset} onSelect={setNotebookDataset} sample={notebookSample} /></div>
        {overview && <Summary models={models} />}
        {(overview || page === 'compare' || page === 'training') && <div className={`work-grid ${overview ? '' : 'single'}`}>
          {(overview || page === 'compare') && <ComparePanel models={models} revision={revision} />}
          {(overview || page === 'training') && <TrainingPanel status={worker || idle} onStart={start}
            versions={datasets?.data?.versions || []} dataset={worker?.running ? worker.dataset : trainingDataset}
            onDataset={setTrainingDataset} />}
        </div>}
        {(overview || page === 'models') && <Deployment revision={revision} onRefresh={refresh} />}
        {(overview || page === 'models') && <Registry models={models} onInspect={setModelKey} onRefresh={refresh} />}
      </div>
    </main>
    {modelKey && <ModelDialog key={modelKey} modelKey={modelKey} onClose={() => setModelKey(null)} />}
  </div>
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>)
