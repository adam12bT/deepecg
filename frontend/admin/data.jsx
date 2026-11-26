import { useState } from 'react'
import { Activity, BarChart3, ChevronDown, Database, FileUp, Layers3, Play, ShieldCheck, Waves } from 'lucide-react'
import { request, useResource } from './api.js'
import { ErrorNote, PanelTitle, Status } from './ui.jsx'
import './data.css'

const number = new Intl.NumberFormat()
const count = (value) => number.format(value)
const sum = (values) => values.reduce((total, value) => total + value, 0)
const titles = { train: 'Training', validation: 'Validation', test: 'Test' }
const clock = (seconds) => [Math.floor(seconds / 3600), Math.floor((seconds / 60) % 60), seconds % 60]
  .map((value) => String(value).padStart(2, '0')).join(':')

function Select({ id, title, value, options, onChange }) {
  return <div className="field"><label htmlFor={id}>{title}</label><div className="select-wrap">
    <select id={id} value={value || ''} onChange={(event) => onChange(event.target.value)}>
      {!options.length && <option value="">Unavailable</option>}
      {options.map(([key, label]) => <option key={key} value={key}>{label}</option>)}
    </select><ChevronDown size={16} />
  </div></div>
}

function DataSummary({ version, splits, stageNames }) {
  const stats = splits.map((split) => version.splits[split])
  const stageEpochs = stats.some((split) => split.stage_counts)
    ? sum(stats.map((split) => sum(split.stage_counts || []))) : null
  const cards = [
    ['Subjects', sum(stats.map((split) => split.subjects)), `${version.splits.train.subjects} in training`, Layers3, 'green'],
    ['Recordings', sum(stats.map((split) => split.recordings)), `${version.splits.train.recordings} in training`, Waves, 'blue'],
    ['Scored epochs', sum(stats.map((split) => sum(split.state_epoch_counts))), '30 seconds per epoch', Activity, 'violet'],
    ['Forecast windows', sum(stats.map((split) => sum(split.forecast_class_counts))), 'Five minutes of EEG each', BarChart3, 'amber'],
    ['Stage-labelled epochs', stageEpochs === null ? '—' : stageEpochs,
      stageEpochs === null ? 'Upload five-class annotations' : stageNames.join(' · '), Layers3, 'blue'],
  ]
  return <section className="metric-grid" aria-label="Dataset overview">
    {cards.map(([label, value, note, Icon, color]) => <div className="metric-card" key={label}>
      <div className="metric-label"><span className={`metric-icon ${color}`}><Icon size={17} /></span>{label}</div>
      <strong>{count(value)}</strong><span className="metric-hint">{note}</span>
    </div>)}
  </section>
}

function SplitOverview({ version, splits }) {
  const hasStages = splits.some((key) => version.splits[key].stage_counts)
  return <section className="panel"><PanelTitle icon={ShieldCheck} title="Subject splits" />
    <div className="table-scroll"><table className="data-table split-table"><thead><tr>
      <th scope="col">Split</th><th scope="col">Subjects</th><th scope="col">Recordings</th>
      <th scope="col">Wake epochs</th><th scope="col">Sleep epochs</th>
      {hasStages && <th scope="col">Scored stage epochs</th>}<th scope="col">Forecast windows</th>
    </tr></thead><tbody>{splits.map((key) => {
      const split = version.splits[key]
      return <tr key={key}><th scope="row">{titles[key]}</th><td>{count(split.subjects)}</td><td>{count(split.recordings)}</td>
        <td>{count(split.state_epoch_counts[0])}</td><td>{count(split.state_epoch_counts[1])}</td>
        {hasStages && <td>{count(sum(split.stage_counts))}</td>}
        <td>{count(sum(split.forecast_class_counts))}</td></tr>
    })}</tbody></table></div>
  </section>
}

function Distributions({ version, splits, labels, stageNames }) {
  const [split, setSplit] = useState('train')
  const stats = version.splits[split]
  const [wake, sleep] = stats.state_epoch_counts
  const max = Math.max(...stats.forecast_class_counts, 1)
  return <section className="panel data-distributions">
    <PanelTitle icon={BarChart3} title="Label distributions" />
    <div className="segment split-filter" aria-label="Label distribution split">{splits.map((key) => <button key={key}
      type="button" aria-pressed={split === key} onClick={() => setSplit(key)}>{titles[key]}</button>)}</div>
    <div className="distribution-subtitle"><h3>Sleep–wake epochs</h3><span>{count(wake + sleep)} total</span></div>
    <div className="label-bar" aria-hidden="true"><span className="wake" style={{ width: `${100 * wake / (wake + sleep)}%` }} />
      <span className="sleep" style={{ width: `${100 * sleep / (wake + sleep)}%` }} /></div>
    <div className="data-legend"><span><i className="wake" />Wake {count(wake)}</span><span><i className="sleep" />Sleep {count(sleep)}</span></div>
    {stats.stage_counts && <><div className="distribution-subtitle"><h3>Sleep stages</h3>
      <span>{count(sum(stats.stage_counts))} scored epochs</span></div>
      <div className="data-legend stage-counts">{stageNames.map((stage, index) =>
        <span key={stage}>{stage} {count(stats.stage_counts[index])}</span>)}</div></>}
    <div className="distribution-subtitle"><h3>Time to sustained awakening</h3><span>{count(sum(stats.forecast_class_counts))} windows</span></div>
    <div className="table-scroll"><div className="forecast-chart" aria-label="Forecast class counts">
      {stats.forecast_class_counts.map((value, index) => <div className="forecast-column" key={labels[index]}
        title={`${labels[index]} · ${count(value)} windows`}>
        <span className="forecast-count">{count(value)}</span>
        <span className={`forecast-stick ${index === labels.length - 1 ? 'capped' : ''}`} style={{ height: `${6 + 82 * value / max}px` }} />
        <span className="forecast-tick">{index === labels.length - 1 ? 'None' : labels[index].replace(' min', '')}</span>
      </div>)}
    </div></div>
    <p className="data-caption">Each bin spans five minutes. None means no qualifying awakening within 60 minutes.</p>
  </section>
}

function Trace({ preview, feature }) {
  const { values, timeSeconds, states, epochSeconds } = preview
  const low = Math.min(...values)
  const high = Math.max(...values)
  const first = timeSeconds[0]
  const last = timeSeconds.at(-1)
  const x = (time) => 80 + 710 * (time - first) / Math.max(last - first, epochSeconds)
  const y = (value) => 153 - 123 * (value - low) / Math.max(high - low, 1e-8)
  const trace = values.map((value, index) => {
    const continuous = index > 0 && timeSeconds[index] - timeSeconds[index - 1] === epochSeconds
    return `${continuous ? 'L' : 'M'}${x(timeSeconds[index]).toFixed(2)},${y(value).toFixed(2)}`
  }).join(' ')
  const ticks = [...new Set([0, Math.floor(values.length / 2), values.length - 1])]
  const barWidth = 710 * epochSeconds / Math.max(last - first + epochSeconds, epochSeconds)
  return <figure className="trace-figure" aria-label={`${feature} values by epoch end`}>
    <svg viewBox="0 0 820 210" className="trace" role="img" aria-label={`EEG feature ${feature}, ${values.length} epochs`}>
      {[30, 91.5, 153].map((height, index) => <g key={height}>
        <line x1="80" y1={height} x2="790" y2={height} className="trace-grid" />
        <text x="69" y={height + 4} textAnchor="end">{[high, (low + high) / 2, low][index].toPrecision(3)}</text>
      </g>)}
      <path d={trace} className="trace-path" />
      {states.map((value, index) => <rect key={timeSeconds[index]} x={x(timeSeconds[index]) - barWidth / 2} y="165"
        width={Math.max(1, barWidth)} height="8" className={value === 1 ? 'trace-sleep' : 'trace-wake'} />)}
      {ticks.map((index) => <text key={index} x={x(timeSeconds[index])} y="195" textAnchor="middle">{clock(timeSeconds[index])}</text>)}
    </svg>
    <figcaption className="data-legend"><span><i className="wake" />Wake</span><span><i className="sleep" />Sleep</span>
      <span>Epoch end from recording start</span></figcaption>
  </figure>
}

function Explorer({ version, splits, features, previewEpochs, onNotebook }) {
  const [split, setSplit] = useState('train')
  const [selected, setSelected] = useState('')
  const [feature, setFeature] = useState(features[0])
  const [start, setStart] = useState(0)
  const base = `/datasets/${version.id}`
  const list = useResource(`${base}/recordings?${new URLSearchParams({ split })}`)
  const records = list?.data?.recordings || []
  const record = records.find((entry) => entry.id === selected)?.id || records[0]?.id
  const result = useResource(record ? `${base}/preview?${new URLSearchParams({ record, feature, start })}` : null)
  const preview = result?.data
  const lastStart = preview ? Math.max(0, preview.total - previewEpochs) : 0

  return <section className="panel"><PanelTitle icon={Waves} title="Recording explorer">
    <button type="button" className="button secondary" disabled={!preview}
      onClick={() => onNotebook({ version: version.id, data: { ...preview, feature } })}>Open in notebook</button>
  </PanelTitle>
    <div className="explorer-controls">
      <Select id="explorer-split" title="Recording split" value={split} options={splits.map((key) => [key, titles[key]])}
        onChange={(key) => { setSplit(key); setSelected(''); setStart(0) }} />
      <Select id="recording" title="Recording" value={record} options={records.map((entry) => [entry.id, entry.id])}
        onChange={(key) => { setSelected(key); setStart(0) }} />
      <Select id="eeg-feature" title="EEG feature" value={feature} options={features.map((key) => [key, key.replaceAll('_', ' ')])}
        onChange={setFeature} />
    </div>
    {list?.error && <ErrorNote>{list.error}</ErrorNote>}
    {result?.error && <ErrorNote>{result.error}</ErrorNote>}
    {!preview && !list?.error && !result?.error && <p className="notice">{list && !records.length ? 'No recordings in this split.' : 'Loading EEG values…'}</p>}
    {preview && <>
      <Trace preview={preview} feature={feature} />
      <div className="trace-controls"><span>Epochs {count(preview.start + 1)}–{count(preview.start + preview.values.length)} of {count(preview.total)}</span>
        <div><button type="button" className="button secondary" disabled={preview.start === 0}
          onClick={() => setStart(Math.max(0, preview.start - previewEpochs))}>Earlier</button>
        <button type="button" className="button secondary" disabled={preview.start >= lastStart}
          onClick={() => setStart(Math.min(lastStart, preview.start + previewEpochs))}>Later</button></div>
      </div>
    </>}
  </section>
}

function UploadData({ version, maxUploadBytes, onSaved }) {
  const [name, setName] = useState('')
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(null)

  async function upload(event) {
    event.preventDefault()
    if (!file || busy) return
    setError('')
    setSaved(null)
    if (!file.name.toLowerCase().endsWith('.npz') || file.size > maxUploadBytes) {
      setError(`Choose a prepared NPZ archive up to ${maxUploadBytes / 1024 ** 2} MiB.`)
      return
    }
    setBusy(true)
    try {
      const body = new FormData()
      body.set('parent', version.id)
      body.set('name', name)
      body.set('file', file)
      const result = await request('/datasets', { method: 'POST', body })
      setSaved(result)
      setName('')
      setFile(null)
      onSaved(result)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setBusy(false)
    }
  }

  return <section className="panel upload-data">
    <PanelTitle icon={FileUp} title="Add training data" />
    <ol className="data-onboarding">
      <li><strong>Download</strong><span>Save this version's archive as your starting point.</span></li>
      <li><strong>Prepare</strong><span>Add new recordings and labels in the project preprocessing workflow.</span></li>
      <li><strong>Upload and train</strong><span>SleepOps validates it, saves a DVC version, then trains a candidate from that version.</span></li>
    </ol>
    <p className="data-caption">Upload a complete prepared <code>.npz</code> feature archive (not raw EDF files). It must use the SleepOps 11-feature schema and include recording IDs, subject IDs, timestamps, labels, and split assignments. New subjects are added to training; existing validation and test subjects stay protected.</p>
    <form onSubmit={upload}>
      <label className="field">Name this dataset version<input id="dataset-name" value={name} maxLength={80}
        placeholder="e.g. Sleep-EDF with new sessions" required onChange={(event) => setName(event.target.value)} disabled={busy} /></label>
      <label className="field">Prepared feature archive<input key={saved?.id || 'file'} id="dataset-upload" type="file" accept=".npz,application/zip"
        required onChange={(event) => setFile(event.target.files[0] || null)} disabled={busy} /></label>
      <button className="button primary" disabled={busy || !file || !name.trim()}><FileUp size={16} />
        {busy ? 'Validating and saving…' : 'Validate and create version'}</button>
    </form>
    <p className="data-caption">Based on {version.name} · maximum upload {maxUploadBytes / 1024 ** 2} MiB</p>
    <a className="data-link" href="https://github.com/adam12bT/deepecg/blob/sleepops/Sleep_EDF_Wake_Up_Prediction_Colab.ipynb"
      target="_blank" rel="noreferrer">Open the EEG preprocessing notebook</a>
    <a className="data-link" href="https://github.com/adam12bT/deepecg/blob/sleepops/Sleep_Stage_Awakening_Training_Colab.ipynb"
      target="_blank" rel="noreferrer">Open the two-stage Colab training notebook</a>
    {error && <ErrorNote>{error}</ErrorNote>}
    {saved && <div className="notice success dataset-saved" role="status"><ShieldCheck size={16} />
      <div><strong>{saved.name} is saved as a private DVC dataset version.</strong>
        <span>Added {count(saved.added.recordings)} recordings and {count(saved.added.stageLabels || 0)} stage labels. Held-out subject splits were preserved.</span>
      </div><button type="button" className="button primary" onClick={() => onTrain(saved.id)}><Play size={15} />Train with this version</button>
    </div>}
  </section>
}

function VersionHistory({ versions, active, onSelect }) {
  return <section className="panel"><PanelTitle icon={Layers3} title="Dataset versions" />
    <div className="table-scroll"><table className="data-table versions-table"><thead><tr>
      <th scope="col">Version</th><th scope="col">Train subjects</th><th scope="col">Train recordings</th>
      <th scope="col">Added recordings</th><th scope="col">Archive SHA-256</th><th scope="col" aria-label="Actions" />
    </tr></thead><tbody>{versions.map((version) => <tr key={version.id}>
      <th scope="row">{version.name} {version.id === 'base' && <Status>Original</Status>}</th>
      <td>{count(version.splits.train.subjects)}</td><td>{count(version.splits.train.recordings)}</td>
      <td>{version.added ? count(version.added.recordings) : '—'}</td>
      <td><code title={version.hash}>{version.hash.slice(0, 12)}</code></td>
      <td><button type="button" className="button inspect-button" onClick={() => onSelect(version.id)}
        aria-label={`View ${version.name}`} aria-pressed={active === version.id}>{active === version.id ? 'Viewing' : 'View'}</button></td>
    </tr>)}</tbody></table></div>
  </section>
}

export function DatasetPage({ resource, selected, onSelect, onSaved, onTrain, onNotebook }) {
  if (!resource) return <p className="notice" role="status">Loading dataset versions…</p>
  if (resource.error) return <ErrorNote>{resource.error}</ErrorNote>
  const { versions, splits, forecastLabels, stageNames, features, previewEpochs, maxUploadBytes } = resource.data
  const version = versions.find((entry) => entry.id === selected) || versions[0]
  return <>
    <section className="panel version-picker">
      <PanelTitle icon={Database} title="Dataset version">
        <div className="data-actions"><a className="button secondary" href={`/api/admin/datasets/${encodeURIComponent(version.id)}/archive`}
          download="sleepops_dataset.npz">Download selected archive</a>
          <a className="button secondary" href={version.url} target="_blank" rel="noreferrer">Open DVC artifacts</a>
          <button type="button" className="button primary" onClick={() => onTrain(version.id)}><Play size={16} /> Train on this version</button></div>
      </PanelTitle>
      <Select id="dataset-version" title="Viewing" value={version.id}
        options={versions.map((entry) => [entry.id, entry.name])} onChange={onSelect} />
    </section>
    <DataSummary version={version} splits={splits} stageNames={stageNames} />
    <div className="data-grid">
      <Distributions version={version} splits={splits} labels={forecastLabels} stageNames={stageNames} />
      <UploadData version={version} maxUploadBytes={maxUploadBytes} onSaved={onSaved} onTrain={onTrain} />
    </div>
    <SplitOverview version={version} splits={splits} />
    <Explorer key={version.id} version={version} splits={splits} features={features} previewEpochs={previewEpochs}
      onNotebook={onNotebook} />
    <VersionHistory versions={versions} active={version.id} onSelect={onSelect} />
  </>
}
