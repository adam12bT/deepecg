import assert from 'node:assert/strict'
import test from 'node:test'
import { parseCsv, parseNpy, readFeatureFile } from './input.js'

const window = Array.from({ length: 10 }, (_, i) => Array.from({ length: 11 }, (_, j) => (i * 11 + j) / 4))

function npy({ littleEndian = true, fortran = false } = {}) {
  const header = new TextEncoder().encode(`{'descr': '${littleEndian ? '<' : '>'}f4', 'fortran_order': ${fortran ? 'True' : 'False'}, 'shape': (10, 11), }\n`)
  const start = header.length + 10
  const bytes = new Uint8Array(start + 440)
  bytes.set([0x93, 78, 85, 77, 80, 89, 1, 0])
  const view = new DataView(bytes.buffer)
  view.setUint16(8, header.length, true)
  bytes.set(header, 10)
  window.forEach((row, i) => row.forEach((value, j) => {
    const index = fortran ? j * 10 + i : i * 11 + j
    view.setFloat32(start + index * 4, value, littleEndian)
  }))
  return bytes.buffer
}

test('loads CSV and NumPy windows in chronological feature order', async () => {
  const csv = window.map((row) => row.join(',')).join('\r\n')
  assert.deepEqual(parseCsv(csv), window)
  assert.deepEqual(await readFeatureFile(new File([csv], 'features.CSV')), window)
  assert.deepEqual(parseNpy(npy()), window)
  assert.deepEqual(parseNpy(npy({ littleEndian: false, fortran: true })), window)
})

test('rejects malformed, nonfinite, or oversized input before inference', async () => {
  assert.throws(() => parseCsv('1,2'), /10 rows/)
  assert.throws(() => parseCsv('column1,column2'), /headerless/)
  assert.throws(() => parseNpy(npy().slice(0, -5)), /data size/)
  await assert.rejects(readFeatureFile(new File(['sample'], 'recording.edf')), /CSV or NumPy/)
  await assert.rejects(readFeatureFile(new File([new Uint8Array(70000)], 'large.csv')), /64 KB/)
})
