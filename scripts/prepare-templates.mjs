/**
 * Rewrites the mission .vn files for the web build and emits the template manifest.
 *
 * The reference graphs are copied straight from the desktop repo, where
 * input_image / input_movie carry absolute paths like
 * /Users/…/docs/Missions/img/galets_atelier.jpg. On the web there is no such
 * filesystem, so each path is remapped onto the matching bundled sample. A file
 * that has no bundled counterpart (the 100 MB Sally.mp4) is left pointing at its
 * bare filename, which the UI shows as "missing" until the student loads it.
 */
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const TEMPLATES_DIR = new URL('../public/templates/', import.meta.url)
const SAMPLES_DIR = new URL('../public/samples/', import.meta.url)

const MISSIONS = [
  {
    file: 'M1.1_reference.vn',
    name: 'TD 1 — Le tamis qui n’existe pas',
    description:
      'Granulométrie de galets sur photo de terrain. Chaîne complète : niveaux de gris, seuillage d’Otsu, morphologie, composantes connexes, mesure des régions et histogramme D10/D50/D90. Calibrée avec une pièce de 2 € (25,75 mm).',
  },
  {
    file: 'M1.2_reference.vn',
    name: 'TD 2 — Le pas de trop',
    description:
      'Morphométrie légale d’une empreinte de pied nu. Masque polygonal, seuillage inverse, redressement par boîte englobante orientée, puis TFL / CBW / HBW et indice de Staheli via la node forensique. Calibrée sur une réglette de 100 mm.',
  },
  {
    file: 'M1.3_reference.vn',
    name: 'TD 3 — Décoder une émotion',
    description:
      'Suivi facial sur vidéo : 478 points MediaPipe, extraction des commissures et des coins externes des yeux, rapport des deux distances (invariant à la distance caméra) tracé image par image. La vidéo Sally.mp4 est à charger manuellement.',
  },
]

const availableSamples = new Set(readdirSync(SAMPLES_DIR))

/** Absolute desktop paths become `samples/<file>` when we ship that file. */
function remapPath(original) {
  const filename = String(original).split('/').pop() ?? ''
  if (availableSamples.has(filename)) return `samples/${filename}`

  // The desktop repo keeps full-resolution images; the web build ships the 900 px
  // variants under the original name, so a `_900` suffix also resolves.
  const withoutSuffix = filename.replace(/_900(\.[a-z0-9]+)$/i, '$1')
  if (availableSamples.has(withoutSuffix)) return `samples/${withoutSuffix}`

  return filename
}

const manifest = []
for (const mission of MISSIONS) {
  const path = join(TEMPLATES_DIR.pathname, mission.file)
  const graph = JSON.parse(readFileSync(path, 'utf8'))

  let rewritten = 0
  const missing = []
  for (const node of graph.nodes ?? []) {
    const params = node.data?.params
    if (!params || typeof params.path !== 'string') continue
    const remapped = remapPath(params.path)
    if (remapped !== params.path) rewritten++
    if (!remapped.startsWith('samples/')) missing.push(remapped)
    params.path = remapped
  }

  writeFileSync(path, JSON.stringify(graph, null, 2))
  manifest.push({ name: mission.name, description: mission.description, file: mission.file })
  const missingNote = missing.length > 0 ? ` — non fournis : ${missing.join(', ')}` : ''
  console.log(`${mission.file}: ${rewritten} chemin(s) remappé(s)${missingNote}`)
}

writeFileSync(join(TEMPLATES_DIR.pathname, 'manifest.json'), JSON.stringify(manifest, null, 2))
console.log(`manifest.json: ${manifest.length} missions`)
