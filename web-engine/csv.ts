/**
 * CSV parsing, close enough to `pandas.read_csv` for the DataFrame nodes.
 *
 * Handles RFC-4180 quoting, the four separators the desktop node offers, and
 * pandas' own dtype inference: a column of whole numbers becomes int64, one
 * with any decimal or any missing value becomes float64, anything else stays
 * object. The missing-value tokens are pandas' default NA list.
 */
import { DataFrame, makeDf } from './dataframe'

/** pandas' default `na_values`, minus the ones that cannot appear unquoted. */
const NA_TOKENS = new Set([
  '', 'na', 'n/a', 'nan', 'null', 'none', '-nan', '-na', '#n/a', '#na',
  '1.#ind', '-1.#ind', '1.#qnan', '-1.#qnan', '<na>', 'nat',
])

/**
 * Splits CSV text into rows of raw fields.
 *
 * Quotes are handled the way pandas does: a quoted field may span newlines, and
 * a doubled quote inside one is a literal quote.
 */
export function parseCsvRows(text: string, separator: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  let started = false

  // A UTF-8 BOM would otherwise become part of the first column's name.
  const body = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text

  for (let i = 0; i < body.length; i++) {
    const ch = body[i]
    if (quoted) {
      if (ch === '"') {
        if (body[i + 1] === '"') { field += '"'; i++ }
        else quoted = false
      } else {
        field += ch
      }
      continue
    }
    if (ch === '"' && field === '') { quoted = true; started = true; continue }
    if (ch === separator) { row.push(field); field = ''; started = true; continue }
    if (ch === '\r') continue
    if (ch === '\n') {
      row.push(field)
      // A trailing newline must not add a phantom row.
      if (row.length > 1 || row[0] !== '' || started) rows.push(row)
      row = []
      field = ''
      started = false
      continue
    }
    field += ch
    started = true
  }
  if (started || field !== '' || row.length > 0) {
    row.push(field)
    rows.push(row)
  }
  return rows
}

function isNaToken(raw: string): boolean {
  return NA_TOKENS.has(raw.trim().toLowerCase())
}

/** Whether the token is a number pandas would parse, and its value. */
function asNumber(raw: string): number | null {
  const trimmed = raw.trim()
  if (trimmed === '') return null
  // Number('') is 0 and Number(' ') is 0, both already excluded above.
  const value = Number(trimmed)
  return Number.isNaN(value) ? null : value
}

/** The literals pandas reads as booleans. */
const TRUE_TOKENS = new Set(['true', 'TRUE', 'True'])
const FALSE_TOKENS = new Set(['false', 'FALSE', 'False'])

function asBoolean(raw: string): boolean | null {
  const trimmed = raw.trim()
  if (TRUE_TOKENS.has(trimmed)) return true
  if (FALSE_TOKENS.has(trimmed)) return false
  return null
}

export interface CsvOptions {
  separator?: string
  /** 0 or absent reads the whole file. */
  maxRows?: number
}

export function parseCsv(text: string, options: CsvOptions = {}): DataFrame {
  const separator = options.separator ?? ','
  const rows = parseCsvRows(text, separator)
  if (rows.length === 0) return makeDf([], [])

  // pandas de-duplicates repeated headers as name, name.1, name.2… and it
  // keeps whatever whitespace the header had, so " b" stays " b".
  const seen = new Map<string, number>()
  const columns = rows[0].map((raw, i) => {
    const base = raw === '' ? `Unnamed: ${i}` : raw
    const count = seen.get(base) ?? 0
    seen.set(base, count + 1)
    return count === 0 ? base : `${base}.${count}`
  })

  const limit = options.maxRows && options.maxRows > 0 ? options.maxRows : Infinity
  const body = rows.slice(1, rows.length).filter((r) => r.length > 1 || (r[0] ?? '') !== '')
  const kept = body.slice(0, limit === Infinity ? undefined : limit)

  // Infer each column's type over the whole column before building the rows,
  // the way pandas does — a single decimal makes the entire column float, and
  // a column of True/False becomes bool rather than text.
  const numeric: boolean[] = columns.map(() => true)
  const boolean: boolean[] = columns.map(() => true)
  const populated: boolean[] = columns.map(() => false)
  for (const raw of kept) {
    columns.forEach((_, c) => {
      const cell = raw[c] ?? ''
      if (isNaToken(cell)) return
      populated[c] = true
      if (numeric[c] && asNumber(cell) === null) numeric[c] = false
      if (boolean[c] && asBoolean(cell) === null) boolean[c] = false
    })
  }

  const records = kept.map((raw) => {
    const record: Record<string, unknown> = {}
    columns.forEach((name, c) => {
      const cell = raw[c] ?? ''
      if (isNaToken(cell)) { record[name] = null; return }
      if (boolean[c] && populated[c]) record[name] = asBoolean(cell)
      else if (numeric[c]) record[name] = asNumber(cell)
      else record[name] = cell
    })
    return record
  })

  return makeDf(columns, records)
}
