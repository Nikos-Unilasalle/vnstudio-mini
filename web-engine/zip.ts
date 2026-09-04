/**
 * Minimal ZIP writer.
 *
 * Export Crops writes one file per detected object. On the desktop those land
 * in a folder; a browser can only hand the user files one at a time, and forty
 * separate download prompts is not an export. So the crops are bundled into a
 * single archive instead.
 *
 * Entries are stored uncompressed (method 0). PNG and JPEG are already
 * compressed, so deflating them again would buy nothing and would mean carrying
 * a compressor.
 */

/** CRC-32 table, built once — the ZIP format requires a checksum per entry. */
const CRC_TABLE = (() => {
  const table = new Uint32Array(256)
  for (let i = 0; i < 256; i++) {
    let c = i
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[i] = c >>> 0
  }
  return table
})()

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

export interface ZipEntry {
  /** Path inside the archive; forward slashes make subfolders. */
  name: string
  bytes: Uint8Array
}

/** Builds a ZIP archive from the given entries. */
export function buildZip(entries: ZipEntry[]): Uint8Array {
  const encoder = new TextEncoder()
  const locals: Uint8Array[] = []
  const centrals: Uint8Array[] = []
  let offset = 0

  for (const entry of entries) {
    const name = encoder.encode(entry.name)
    const checksum = crc32(entry.bytes)
    const size = entry.bytes.length

    const local = new Uint8Array(30 + name.length)
    const lv = new DataView(local.buffer)
    lv.setUint32(0, 0x04034b50, true) // local file header
    lv.setUint16(4, 20, true) // version needed
    lv.setUint16(6, 0, true) // flags
    lv.setUint16(8, 0, true) // method: stored
    lv.setUint16(10, 0, true) // mod time
    lv.setUint16(12, 0, true) // mod date
    lv.setUint32(14, checksum, true)
    lv.setUint32(18, size, true) // compressed size
    lv.setUint32(22, size, true) // uncompressed size
    lv.setUint16(26, name.length, true)
    lv.setUint16(28, 0, true) // extra length
    local.set(name, 30)

    const central = new Uint8Array(46 + name.length)
    const cv = new DataView(central.buffer)
    cv.setUint32(0, 0x02014b50, true) // central directory header
    cv.setUint16(4, 20, true) // version made by
    cv.setUint16(6, 20, true) // version needed
    cv.setUint16(8, 0, true)
    cv.setUint16(10, 0, true)
    cv.setUint16(12, 0, true)
    cv.setUint16(14, 0, true)
    cv.setUint32(16, checksum, true)
    cv.setUint32(20, size, true)
    cv.setUint32(24, size, true)
    cv.setUint16(28, name.length, true)
    cv.setUint16(30, 0, true) // extra
    cv.setUint16(32, 0, true) // comment
    cv.setUint16(34, 0, true) // disk number
    cv.setUint16(36, 0, true) // internal attrs
    cv.setUint32(38, 0, true) // external attrs
    cv.setUint32(42, offset, true) // offset of local header
    central.set(name, 46)

    locals.push(local, entry.bytes)
    centrals.push(central)
    offset += local.length + size
  }

  const centralSize = centrals.reduce((n, c) => n + c.length, 0)
  const end = new Uint8Array(22)
  const ev = new DataView(end.buffer)
  ev.setUint32(0, 0x06054b50, true) // end of central directory
  ev.setUint16(4, 0, true) // disk
  ev.setUint16(6, 0, true) // disk with central dir
  ev.setUint16(8, entries.length, true)
  ev.setUint16(10, entries.length, true)
  ev.setUint32(12, centralSize, true)
  ev.setUint32(16, offset, true) // central dir offset
  ev.setUint16(20, 0, true) // comment length

  const total = offset + centralSize + end.length
  const out = new Uint8Array(total)
  let at = 0
  for (const chunk of [...locals, ...centrals, end]) {
    out.set(chunk, at)
    at += chunk.length
  }
  return out
}
