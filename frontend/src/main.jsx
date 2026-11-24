import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, AudioLines, ChevronDown, CircleAlert, Clock3, Database, FileUp,
  FlaskConical, MoonStar, Radar, Sun, TimerReset, Upload, Waves,
} from 'lucide-react'
import '@fontsource-variable/inter'
import '@fontsource-variable/space-grotesk'
import { readFeatureFile } from './input.js'
import './styles.css'

const empty = { status: 'idle' }
const pct = (value) => `${(100 * value).toFixed(1)}%`

function clock(seconds) {
  return [Math.floor(seconds / 3600), Math.floor((seconds % 3600) / 60), seconds % 60]
    .map((number) => String(number).padStart(2, '0')).join(':')
}

function offset(index, count, epochSeconds) {
  const seconds = (count - 1 - index) * epochSeconds
  if (seconds === 0) return 'Now'
  return `−${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}

async function api(path, options) {
  const response = await fetch(`/api${path}`, options)
  const body = await response.json().catch(() => null)
  if (!response.ok || !body) {
    const detail = body?.detail
    throw new Error(typeof detail === 'string' ? detail : 'The inference service could not complete this request.')
  }
  return body
}

function Banner({ children, retry }) {
  return <div className="error-banner" role="alert">
    <CircleAlert size={18} aria-hidden="true" /><span>{children}</span>
    {retry && <button className="text-button" onClick={retry}>Retry</button>}
  </div>
}

function Select({ id, label, value, change, options, disabled = false }) {
  return <div className="field">
    <label htmlFor={id}>{label}</label>
    <div className="select-wrap">
      <select id={id} value={value} onChange={(event) => change(event.target.value)} disabled={disabled}>
        {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
      </select>
      <ChevronDown size={17} aria-hidden="true" />
    </div>
  </div>
}

function RecordingControls({ recordings, record, end, index, setIndex, choose, historySeconds, isReady }) {
  const ends = record?.window_ends_sec ?? []
  if (!isReady) return <p className="muted">Loading held-out recordings…</p>
  if (!ends.length) return <p className="muted">No held-out windows are available. You can still upload an EEG window.</p>
  return <div className="recording-controls">
    <Select id="recording" label="Held-out recording" value={record.id} change={choose}
      options={recordings.map((item) => [item.id, item.id])} />
    <Select id="window" label="Window end from recording start" value={index} change={(value) => setIndex(Number(value))}
      options={ends.map((seconds, i) => [i, clock(seconds)])} />
    <div className="window-nav">
      <button disabled={index === 0} onClick={() => setIndex(index - 1)}>Previous</button>
      <button disabled={index >= ends.length - 1} onClick={() => setIndex(index + 1)}>Next window</button>
    </div>
    <label className="sr-only" htmlFor="window-position">Window position</label>
    <input id="window-position" className="scrubber" type="range" min="0" max={ends.length - 1} value={index}
      aria-valuetext={clock(end)} onChange={(event) => setIndex(Number(event.target.value))} />
    <div className="scrubber-labels"><span>{clock(ends[0])}</span><span>{clock(ends.at(-1))}</span></div>
    <div className="window-range">
      <Clock3 size={18} aria-hidden="true" />
      <div><span>EEG used for this prediction</span><strong>{clock(end - historySeconds)} – {clock(end)}</strong></div>
    </div>
  </div>
}

function UploadControls({ schema, request, upload, ready }) {
  const input = useRef(null)
  const [dragging, setDragging] = useState(false)
  return <div className="upload-controls">
    <button className={`drop-zone${dragging ? ' dragging' : ''}`} onClick={() => input.current.click()} disabled={!ready}
      onDragOver={(event) => { event.preventDefault(); if (ready) setDragging(true) }}
      onDragLeave={() => setDragging(false)} onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        if (ready && event.dataTransfer.files[0]) upload(event.dataTransfer.files[0])
      }}>
      <FileUp size={25} aria-hidden="true" />
      <strong>{request.fileName || 'Choose an EEG window'}</strong>
      <span>{request.status === 'loading' ? 'Analyzing window…' : 'Drop a CSV or NumPy file here'}</span>
    </button>
    <input ref={input} type="file" accept=".csv,.npy" aria-label="Upload an EEG window" hidden
      onChange={(event) => {
        const file = event.target.files[0]
        event.target.value = ''
        if (file) upload(file)
      }} />
    <p className="muted">Use 10 rows of 11 unscaled EEG features in chronological order, with the most recent epoch last.</p>
    {schema && <details>
      <summary>Feature column order</summary>
      <ol className="feature-list">{schema.feature_names.map((name) => <li key={name}>{name}</li>)}</ol>
    </details>}
  </div>
}

function Stat({ label, value, note, icon: Icon, tone = 'neutral', children }) {
  return <section className={`stat panel ${tone}`}>
    <div className="stat-top"><span>{label}</span><div className="stat-icon"><Icon size={19} aria-hidden="true" /></div></div>
    <div className="stat-value">{value}</div>
    <p>{note}</p>
    {children}
  </section>
}

function PanelTitle({ icon: Icon, title, description }) {
  return <div className="panel-title">
    <div className="panel-icon"><Icon size={19} aria-hidden="true" /></div>
    <div><h2>{title}</h2>{description && <p>{description}</p>}</div>
  </div>
}

function Timeline({ result, epochSeconds, status }) {
  const probabilities = result?.asleep_probabilities ?? []
  const [inspected, inspect] = useState(probabilities.length - 1)
  const index = inspected >= 0 ? inspected : probabilities.length - 1
  const threshold = result?.state_threshold ?? 0.5
  return <section className="panel chart-panel" aria-labelledby="timeline-title">
    <div className="panel-head">
      <div className="panel-title">
        <div className="panel-icon"><Activity size={19} aria-hidden="true" /></div>
        <div><h2 id="timeline-title">Sleep–wake timeline</h2><p>Sleep probability for each {epochSeconds}-second epoch</p></div>
      </div>
      <div className="legend"><span><i className="legend-sleep" />Asleep</span><span><i className="legend-wake" />Awake</span></div>
    </div>
    {probabilities.length ? <>
      <div className="chart" style={{ '--epochs': probabilities.length }}>
        <div className="y-labels">{[100, 75, 50, 25, 0].map((label) => <span key={label}>{label}%</span>)}</div>
        <div className="plot" role="group" aria-label="Epoch sleep probabilities">
          <div className="threshold" aria-hidden="true" style={{ bottom: `${threshold * 100}%` }} />
          {probabilities.map((value, i) => <button key={i} className="epoch" aria-pressed={i === index}
            onMouseEnter={() => inspect(i)} onFocus={() => inspect(i)} onClick={() => inspect(i)}
            aria-label={`${offset(i, probabilities.length, epochSeconds)}, sleep probability ${pct(value)}`}>
            <span className={`epoch-fill ${value >= threshold ? 'sleep' : 'wake'}`} style={{ height: `${value * 100}%` }} />
          </button>)}
        </div>
        <div className="axis-caption" />
        <div className="x-labels" aria-hidden="true">
          {probabilities.map((_, i) => <span key={i} data-edge={i === 0 || i === probabilities.length - 1}>
            {offset(i, probabilities.length, epochSeconds)}
          </span>)}
        </div>
      </div>
      <div className="chart-foot"><span><i className="threshold-dash" />Sleep threshold {(threshold * 100).toFixed(0)}%</span>
        <span><strong>{offset(index, probabilities.length, epochSeconds)}</strong> · {pct(probabilities[index])} sleep</span></div>
    </> : <div className="chart-placeholder"><AudioLines size={30} aria-hidden="true" />
      <p>{status === 'loading' ? 'Analyzing EEG window…' : 'Select a recording or upload a window to see its timeline.'}</p>
    </div>}
  </section>
}

function Forecast({ result, status }) {
  const forecast = result?.forecast
  const best = forecast?.bins.reduce((left, right) => right.probability > left.probability ? right : left)
  const mostLikely = best === forecast?.bins.at(-1) ? 'No sustained awakening within 60 minutes' : best?.interval
  return <section className="panel forecast-panel">
    <div className="panel-head"><PanelTitle icon={TimerReset} title="Awakening forecast" description="Probability by time until sustained awakening" /></div>
    {forecast ? <>
      <div className="likely"><span>Most likely individual time bin</span><strong>{mostLikely}</strong></div>
      <div className="forecast-grid">
        {forecast.bins.map((bin, i) => <div key={bin.interval} className={`forecast-bin${i === forecast.bins.length - 1 ? ' horizon' : ''}`}>
          <div className="bin-label"><span>{i === forecast.bins.length - 1 ? 'No awakening within 60 min' : bin.interval}</span>
            <strong>{pct(bin.probability)}</strong></div>
          <div className="bin-track" aria-hidden="true"><span style={{ width: `${bin.probability * 100}%` }} /></div>
        </div>)}
      </div>
    </> : <div className="forecast-empty"><MoonStar size={21} aria-hidden="true" />
      <span>{result ? 'Forecasts appear when the model predicts that the person is asleep.' :
        status === 'loading' ? 'Analyzing EEG window…' : 'Select an EEG window to start.'}</span>
    </div>}
  </section>
}

function App() {
  const [source, setSource] = useState('recording')
  const [context, setContext] = useState(null)
  const [contextError, setContextError] = useState('')
  const [contextRevision, refreshContext] = useState(0)
  const [recordId, setRecordId] = useState('')
  const [index, setIndex] = useState(0)
  const [request, setRequest] = useState(empty)
  const [uploadRequest, setUploadRequest] = useState(empty)
  const [requestRevision, retryRecording] = useState(0)
  const uploadController = useRef(null)

  useEffect(() => {
    const controller = new AbortController()
    const options = { signal: controller.signal }
    Promise.all([api('/model', options), api('/v1/schema', options), api('/v1/demo/recordings', options)])
      .then(([model, schema, catalog]) => {
        if (controller.signal.aborted) return
        setContext({ model, schema, recordings: catalog.recordings })
        const first = catalog.recordings[0]
        if (first) {
          setRecordId(first.id)
          setIndex(Math.floor(first.window_ends_sec.length / 2))
        }
      })
      .catch((error) => { if (!controller.signal.aborted) setContextError(error.message) })
    return () => controller.abort()
  }, [contextRevision])

  const recordings = context?.recordings ?? []
  const record = recordings.find((item) => item.id === recordId)
  const end = record?.window_ends_sec[index]
  const key = end === undefined ? '' : `${recordId}/${end}`
  const epochSeconds = context?.schema.epoch_seconds ?? 30
  const historySeconds = (context?.schema.shape[0] ?? 10) * epochSeconds

  useEffect(() => {
    if (source !== 'recording' || !key) return undefined
    const controller = new AbortController()
    const timer = setTimeout(() => {
      setRequest((previous) => previous.key === key && previous.result ? previous : { status: 'loading', key })
      api(`/v1/demo/recordings/${encodeURIComponent(recordId)}/${end}/prediction`, { signal: controller.signal })
        .then((result) => { if (!controller.signal.aborted) setRequest({ status: 'ready', key, result }) })
        .catch((error) => { if (!controller.signal.aborted) setRequest({ status: 'error', key, error: error.message }) })
    }, 130)
    return () => { clearTimeout(timer); controller.abort() }
  }, [source, recordId, end, key, requestRevision])

  useEffect(() => () => uploadController.current?.abort(), [])

  async function upload(file) {
    uploadController.current?.abort()
    const controller = new AbortController()
    uploadController.current = controller
    const fileName = file.name
    setUploadRequest({ status: 'loading', fileName })
    try {
      const features = await readFeatureFile(file)
      if (controller.signal.aborted) return
      const result = await api('/v1/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        signal: controller.signal, body: JSON.stringify({ features }) })
      if (!controller.signal.aborted) setUploadRequest({ status: 'ready', fileName, result })
    } catch (error) {
      if (!controller.signal.aborted) setUploadRequest({ status: 'error', fileName, error: error.message })
    }
  }

  function choose(id) {
    const selected = recordings.find((item) => item.id === id)
    setRecordId(id)
    setIndex(Math.floor(selected.window_ends_sec.length / 2))
  }

  const current = source === 'upload' ? uploadRequest : request.key === key ? request : { status: key ? 'loading' : 'idle' }
  const result = current.result
  const asleep = result?.state === 'asleep'
  const stateName = result ? (asleep ? 'Asleep' : 'Awake') : current.status === 'loading' ? 'Analyzing…' : '—'
  const modelId = result?.release_id ?? context?.model.release_id
  return <>
    <header className="topbar">
      <div className="topbar-inner">
        <div className="brand"><span className="brand-mark"><Waves size={23} aria-hidden="true" /></span>
          <span>SleepOps</span><span className="brand-divider" /><span className="product-label">EEG analysis</span></div>
        <div className={`connection ${context ? 'online' : ''}`} role="status"><i />{context ? 'Model online' : contextError ? 'Unavailable' : 'Connecting'}</div>
      </div>
    </header>
    <main className="shell">
      <div className="page-heading">
        <div><h1>Sleep &amp; wake analysis</h1><p>Explore sleep state and awakening probabilities from EEG.</p></div>
        {modelId && <div className="model-tag" title={modelId}><FlaskConical size={17} aria-hidden="true" /><span>Model <code>{modelId.slice(0, 12)}</code></span></div>}
      </div>
      {contextError && <Banner retry={() => { setContextError(''); refreshContext((value) => value + 1) }}>{contextError}</Banner>}
      <div className="dashboard">
        <aside className="sidebar">
          <section className="panel input-panel">
            <PanelTitle icon={Waves} title="EEG input" />
            <div className="mode-switch" role="group" aria-label="EEG source">
              <button aria-pressed={source === 'recording'} onClick={() => setSource('recording')}><Database size={16} aria-hidden="true" />Recording</button>
              <button aria-pressed={source === 'upload'} onClick={() => setSource('upload')}><Upload size={16} aria-hidden="true" />Upload</button>
            </div>
            {source === 'recording' ? <RecordingControls recordings={recordings} record={record} index={index} end={end}
              setIndex={setIndex} choose={choose} historySeconds={historySeconds} isReady={!!context || !!contextError} /> :
              <UploadControls schema={context?.schema} request={uploadRequest} upload={upload} ready={!!context} />}
            {current.error && <Banner retry={source === 'recording' ? () => retryRecording((value) => value + 1) : null}>{current.error}</Banner>}
          </section>
          <section className="panel input-spec">
            <div className="spec-heading"><AudioLines size={18} aria-hidden="true" /><h2>Window format</h2></div>
            <dl><div><dt>History</dt><dd>{historySeconds / 60} minutes</dd></div>
              <div><dt>Epoch duration</dt><dd>{epochSeconds} seconds</dd></div>
              <div><dt>Feature matrix</dt><dd>{context?.schema.shape.join(' × ') ?? '10 × 11'}</dd></div></dl>
          </section>
        </aside>
        <div className="analysis">
          <div className="stats" aria-live="polite">
            <Stat label="Current state" value={stateName} icon={asleep ? MoonStar : result ? Sun : Activity}
              tone={asleep ? 'sleep' : result ? 'wake' : 'neutral'} note={result ? `${pct(result.asleep_probabilities.at(-1))} sleep probability` : 'At the final epoch'}>
              {result && <div className="stat-track" aria-hidden="true"><span style={{ width: `${result.asleep_probabilities.at(-1) * 100}%` }} /></div>}
            </Stat>
            <Stat label="Window end" value={source === 'recording' && end !== undefined ? clock(end) : source === 'upload' && result ? 'Uploaded' : '—'}
              icon={Clock3} note={source === 'recording' ? 'Elapsed from recording start' : 'Most recent epoch in the file'} tone="indigo" />
            <Stat label="Awakening within 60 min" value={result?.forecast ? pct(result.forecast.within_hour_probability) : '—'}
              icon={Radar} note={result?.forecast ? 'Sustained awakening' : 'Available during sleep'} tone="teal" />
          </div>
          <Timeline key={source === 'recording' ? key : uploadRequest.fileName ?? 'upload'} result={result} epochSeconds={epochSeconds} status={current.status} />
          <Forecast result={result} status={current.status} />
        </div>
      </div>
    </main>
  </>
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>)
