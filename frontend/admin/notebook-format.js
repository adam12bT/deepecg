export const maxNotebookBytes = 2 * 1024 * 1024
export const maxCells = 80

export function newCell(source = '', kind = 'code') {
  return { id: crypto.randomUUID(), kind, source }
}

export function starterNotebook() {
  return {
    title: 'SleepOps analysis',
    cells: [
      newCell('if dataset is None:\n    print("Run this cell again once the dataset version is visible.")\nelse:\n    print(dataset["name"])\n    for split, counts in dataset["splits"].items():\n        wake, sleep = counts["state_epoch_counts"]\n        print(f"{split:>10}  {counts[\'subjects\']} subjects  |  {wake + sleep:,} scored epochs")'),
      newCell('from statistics import mean\n\nif recording is None:\n    print("Send a recording window from the Data tab.")\nelse:\n    print(recording["id"], recording["feature"])\n    print(f"Mean feature value: {mean(recording[\'values\']):.3f}")\n    print(f"Sleep: {sum(recording[\'states\']) * recording[\'epochSeconds\'] / 60:.1f} minutes")'),
    ],
  }
}

export function parseNotebook(text, title) {
  const doc = JSON.parse(text)
  if (doc.nbformat !== 4 || !Array.isArray(doc.cells) || doc.cells.length > maxCells) {
    throw new Error(`Choose a Jupyter v4 notebook with up to ${maxCells} cells.`)
  }
  const language = doc.metadata?.language_info?.name || doc.metadata?.kernelspec?.language
  if (language && language.toLowerCase() !== 'python') throw new Error('Choose a Python notebook.')
  const cells = doc.cells.map((cell) => {
    const source = Array.isArray(cell.source) && cell.source.every((line) => typeof line === 'string')
      ? cell.source.join('') : cell.source
    if (!['code', 'markdown', 'raw'].includes(cell.cell_type) || typeof source !== 'string') {
      throw new Error('The notebook contains an unsupported cell.')
    }
    return newCell(source, cell.cell_type)
  })
  return { title, cells: cells.length ? cells : [newCell()] }
}

function cellOutputs(output) {
  if (!output) return []
  const outputs = (output.logs || []).map(({ kind, text }) => ({ output_type: 'stream', name: kind, text }))
  if (output.result) outputs.push({ output_type: 'execute_result', execution_count: output.sequence,
    data: { 'text/plain': output.result }, metadata: {} })
  for (const image of output.images || []) outputs.push({ output_type: 'display_data',
    data: { 'image/png': image }, metadata: {} })
  if (output.error) outputs.push({ output_type: 'error', ename: 'PythonError', evalue: output.error.split('\n').filter(Boolean).at(-1),
    traceback: output.error.split('\n') })
  return outputs
}

export function serializeNotebook(notebook, results) {
  return JSON.stringify({ nbformat: 4, nbformat_minor: 5,
    metadata: { kernelspec: { name: 'python3', display_name: 'Python 3', language: 'python' },
      language_info: { name: 'python' } },
    cells: notebook.cells.map(({ id, kind, source }) => {
      const entry = { id, cell_type: kind, metadata: {}, source }
      return kind === 'code' ? { ...entry, execution_count: results[id]?.sequence ?? null,
        outputs: cellOutputs(results[id]) } : entry
    }),
  }, null, 2)
}
