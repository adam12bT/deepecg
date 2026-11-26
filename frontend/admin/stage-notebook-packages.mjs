import { createHash } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

const [directory, version] = process.argv.slice(2)
const { packages } = JSON.parse(await readFile(join(directory, 'pyodide-lock.json'), 'utf8'))
const included = new Set()

function add(name) {
  if (included.has(name)) return
  included.add(name)
  for (const dependency of packages[name].depends) add(dependency)
}

for (const name of ['numpy', 'pandas', 'matplotlib', 'micropip']) add(name)
for (const name of included) {
  const item = packages[name]
  const response = await fetch(new URL(item.file_name, `https://cdn.jsdelivr.net/pyodide/v${version}/full/`), {
    signal: AbortSignal.timeout(90000),
  })
  if (!response.ok) throw new Error(`Could not download ${name} (${response.status}).`)
  const bytes = Buffer.from(await response.arrayBuffer())
  if (createHash('sha256').update(bytes).digest('hex') !== item.sha256) {
    throw new Error(`The Pyodide lock checksum did not match for ${name}.`)
  }
  await writeFile(join(directory, item.file_name), bytes)
}
console.log(`Bundled ${included.size} verified Python packages.`)
