const ROWS = 10
const COLUMNS = 11
const MAX_BYTES = 64 * 1024
const BAD_SHAPE = 'Expected 10 rows with 11 features in each row.'

function checkWindow(rows) {
  if (rows.length !== ROWS || rows.some((row) => row.length !== COLUMNS)) {
    throw new Error(BAD_SHAPE)
  }
  if (rows.some((row) => row.some((value) => !Number.isFinite(value) || Math.abs(value) > 1e6))) {
    throw new Error('Every feature must be a finite number between −1,000,000 and 1,000,000.')
  }
  return rows
}

export function parseCsv(text) {
  const number = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i
  return checkWindow(text.trim().split(/\r?\n/).map((line) => line.split(',').map((cell) => {
    const text = cell.trim()
    if (!number.test(text)) throw new Error('Use a headerless CSV with numeric features.')
    return Number(text)
  })))
}

export function parseNpy(buffer) {
  const bytes = new Uint8Array(buffer)
  const decoder = new TextDecoder()
  if (bytes.length < 10 || bytes[0] !== 0x93 || decoder.decode(bytes.subarray(1, 6)) !== 'NUMPY') {
    throw new Error('This file is not a valid NumPy array.')
  }
  const view = new DataView(buffer)
  const version = bytes[6]
  if (![1, 2, 3].includes(version) || (version > 1 && bytes.length < 12)) {
    throw new Error('Unsupported NumPy array format.')
  }
  const prefix = version === 1 ? 10 : 12
  const headerLength = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
  const start = prefix + headerLength
  if (start >= bytes.length) throw new Error('The NumPy file is incomplete.')
  const header = decoder.decode(bytes.subarray(prefix, start))
  const shape = header.match(/['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)/)
  if (!shape || Number(shape[1]) !== ROWS || Number(shape[2]) !== COLUMNS) throw new Error(BAD_SHAPE)
  const format = header.match(/['"]descr['"]\s*:\s*['"]([<>=|])([fiu])([1248])['"]/)
  const order = header.match(/['"]fortran_order['"]\s*:\s*(True|False)/)
  const readTypes = { f4: 'getFloat32', f8: 'getFloat64', i1: 'getInt8', i2: 'getInt16', i4: 'getInt32',
    i8: 'getBigInt64', u1: 'getUint8', u2: 'getUint16', u4: 'getUint32', u8: 'getBigUint64' }
  const reader = format && readTypes[format[2] + format[3]]
  if (!reader || !order || (format[1] === '|' && format[3] !== '1')) {
    throw new Error('Use a NumPy array of real numbers, such as float32.')
  }
  const width = Number(format[3])
  if (start + width * ROWS * COLUMNS !== bytes.length) throw new Error('The NumPy data size does not match its header.')
  const littleEndian = format[1] !== '>'
  return checkWindow(Array.from({ length: ROWS }, (_, row) => Array.from({ length: COLUMNS }, (_, column) => {
    const index = order[1] === 'True' ? column * ROWS + row : row * COLUMNS + column
    return Number(view[reader](start + index * width, littleEndian))
  })))
}

export async function readFeatureFile(file) {
  if (file.size > MAX_BYTES) throw new Error('Choose a single EEG window smaller than 64 KB.')
  if (/\.csv$/i.test(file.name)) return parseCsv(await file.text())
  if (/\.npy$/i.test(file.name)) return parseNpy(await file.arrayBuffer())
  throw new Error('Choose a CSV or NumPy (.npy) file.')
}
