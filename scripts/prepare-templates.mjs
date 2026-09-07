/**
 * Rewrites the bundled .vn graphs for the web build and emits the two manifests.
 *
 * The graphs are copied straight from the desktop repo, where input_image /
 * input_movie carry absolute paths like /Users/…/samples/billes.jpg. On the web
 * there is no such filesystem, so each path is remapped onto the matching
 * bundled sample; a file with no counterpart keeps its bare name, which the UI
 * shows as "missing" until the user loads one.
 *
 * Two menus are produced. `templates/` holds the general demos. `classes/`
 * holds the course material, grouped under section headings, so the teaching
 * graphs stay available without the front page reading like a syllabus.
 */
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const TEMPLATES_DIR = new URL('../public/templates/', import.meta.url)
const CLASSES_DIR = new URL('../public/classes/', import.meta.url)
const SAMPLES_DIR = new URL('../public/samples/', import.meta.url)

/** The general demos, in the order the menu shows them. */
const DEMOS = [
  {
    file: 'galets_segmenter.vn',
    name: 'Segmentation de galets',
    description:
      'Chaîne de segmentation complète sur une photo de galets : ROI polygonale, redressement en perspective, seuillage adaptatif, morphologie, transformée de distance et watershed pour séparer les galets qui se touchent.',
  },
  {
    file: 'granulometrie_optique.vn',
    name: 'Granulométrie optique',
    description:
      'Mesure la distribution de taille d’une population de billes : CLAHE, seuillage, watershed, puis statistiques et courbe granulométrique. La version longue de la segmentation, avec l’analyse au bout.',
  },
  {
    file: 'sphere_counting_hough.vn',
    name: 'Comptage par transformée de Hough',
    description:
      'Détecte et compte des billes par cercles de Hough plutôt que par seuillage — robuste aux objets qui se touchent. Regroupement des rayons en classes par k-means et rapport d’analyse.',
  },
  {
    file: 'region_props_analysis.vn',
    name: 'Propriétés des régions',
    description:
      'Extrait aire, périmètre, circularité et excentricité de chaque région détectée, les affiche en carte de chaleur et les exporte en CSV. Avec barre d’échelle.',
  },
  {
    file: 'reaction_diffusion.vn',
    name: 'Réaction-diffusion',
    description:
      'Le modèle de Gray-Scott, qui fait émerger des motifs organiques à partir de deux réactifs. Le résultat est seuillé, colorisé et fondu sur une photo.',
  },
  {
    file: 'fft_debruitage_frequentiel.vn',
    name: 'Débruitage fréquentiel',
    description:
      'Passe le flux webcam en Fourier, atténue les hautes fréquences où vit le bruit, et reconstruit. Le spectre est visible pendant qu’on règle le gain.',
  },
  {
    file: 'fft_sharpening_adaptatif.vn',
    name: 'Accentuation adaptative',
    description:
      'Le même trajet aller-retour en Fourier, mais en amplifiant les hautes fréquences : une netteté qui se règle bande par bande plutôt qu’avec un simple masque flou.',
  },
  {
    file: 'fft_egaliseur_spectral.vn',
    name: 'Égaliseur spectral',
    description:
      'Un égaliseur à trois bandes pour l’image : basses, moyennes et hautes fréquences se dosent séparément, comme sur une chaîne hi-fi.',
  },
  {
    file: 'harris_corner.vn',
    name: 'Coins de Harris',
    description:
      'Détecte les coins d’un mur de pierre, les convertit en masque, et les compose sur un fond travaillé en ombres/hautes lumières.',
  },
  {
    file: 'feature_matching.vn',
    name: 'Appariement de points d’intérêt',
    description:
      'Détecte des points ORB sur une photo et sur une copie tournée de 20°, puis trace les correspondances : la démonstration que les descripteurs ORB résistent à la rotation. Remplacez les deux sources par vos propres photos d’une même scène.',
  },
  {
    file: 'smile_detector.vn',
    name: 'Détecteur de sourire',
    description:
      'Suivi facial MediaPipe sur webcam : le rapport entre l’écartement des commissures et celui des yeux — invariant à la distance caméra — franchit un seuil et déclenche l’enregistrement.',
  },
  {
    file: 'magic_painter.vn',
    name: 'Peinture gestuelle',
    description:
      'Suivi de la main sur webcam : l’index devient un pinceau et trace sur une surcouche transparente. Un bon premier contact avec les nodes de geste.',
  },
  {
    file: 'evm_pulse.vn',
    name: 'Pouls par amplification',
    description:
      'Amplification eulérienne du mouvement : les variations de couleur du visage, invisibles à l’œil, sont amplifiées jusqu’à faire apparaître le pouls, puis lissées par filtre de Kalman.',
  },
  {
    file: 'ndvi_ground_truth.vn',
    name: 'Peintre d’indices',
    description:
      'Dessinez des zones et affectez-leur une valeur d’indice (style NDVI) pour fabriquer une vérité terrain synthétique : colorisation, histogramme et propriétés des régions suivent.',
  },
]

/** The course material, grouped by the heading its menu shows it under. */
const CLASSES = [
  {
    section: 'Machine Learning',
    entries: [
      { file: 'TP01_charger_explorer.vn', name: 'Charger et explorer', description: 'Premier contact avec un jeu de données : chargement, aperçu, types de colonnes et statistiques descriptives.' },
      { file: 'TP02_selection_filtrage.vn', name: 'Sélection et filtrage', description: 'Restreindre un tableau aux lignes et colonnes utiles, et lire l’effet de chaque filtre sur la distribution.' },
      { file: 'TP03_nettoyage.vn', name: 'Nettoyage des données', description: 'Valeurs manquantes, doublons et aberrants : les repérer, décider quoi en faire, mesurer ce que ça change.' },
      { file: 'TP04_transformation.vn', name: 'Transformation', description: 'Créer des colonnes dérivées, normaliser, encoder — préparer un tableau pour un modèle.' },
      { file: 'TP05_groupby_agregation.vn', name: 'Groupement et agrégation', description: 'Regrouper par catégorie et résumer : moyennes, comptages et comparaisons entre groupes.' },
      { file: '01_exploration_des_donnees.vn', name: 'Exploration des données', description: 'Parcours guidé d’un jeu de données : formes, types, valeurs manquantes et premières visualisations.' },
      { file: '02_filtrage_et_correlations.vn', name: 'Filtrage et corrélations', description: 'Matrice de corrélation (Pearson, Spearman, Kendall) et lecture des dépendances entre variables.' },
      { file: '03_knn_classificateur.vn', name: 'Classification par k-NN', description: 'k plus proches voisins sur iris : frontière de décision, effet de k et de la métrique, rapport de classification.' },
      { file: '04_clustering_kmeans_pca.vn', name: 'Clustering k-means et ACP', description: 'Partition non supervisée par k-means, projection ACP pour la visualiser, inertie et score de silhouette.' },
      { file: '05_regression_lineaire.vn', name: 'Régression linéaire', description: 'Ajustement, R², RMSE, résidus et coefficients — sur le jeu diabetes.' },
      { file: '06_comparaison_modeles.vn', name: 'Comparaison de modèles', description: 'Même découpage train/test pour plusieurs modèles, et lecture comparée de leurs performances.' },
      { file: '07_random_forest.vn', name: 'Forêt aléatoire', description: 'Ensemble d’arbres : score OOB, importance des variables et matrice de confusion.' },
    ],
  },
  {
    section: 'Computer Vision',
    entries: [
      {
        file: 'M1.1_reference.vn',
        name: 'Le tamis qui n’existe pas',
        description:
          'Granulométrie de galets sur photo de terrain. Chaîne complète : niveaux de gris, seuillage d’Otsu, morphologie, composantes connexes, mesure des régions et histogramme D10/D50/D90. Calibrée avec une pièce de 2 € (25,75 mm).',
      },
      {
        file: 'M1.2_reference.vn',
        name: 'Le pas de trop',
        description:
          'Morphométrie légale d’une empreinte de pied nu. Masque polygonal, seuillage inverse, redressement par boîte englobante orientée, puis TFL / CBW / HBW et indice de Staheli. Calibrée sur une réglette de 100 mm.',
      },
      {
        file: 'M1.3_reference.vn',
        name: 'Décoder une émotion',
        description:
          'Suivi facial sur vidéo : 478 points MediaPipe, extraction des commissures et des coins externes des yeux, rapport des deux distances tracé image par image. La vidéo est à charger manuellement.',
      },
    ],
  },
]

/**
 * Small edits that make a demo work on the web, applied after the path remap.
 *
 * Feature matching shipped with two empty image slots, so both fell back to the
 * schema default and ORB matched a picture against itself. Pointing the second
 * slot through a rotation makes it demonstrate what ORB is actually for —
 * matching across a viewpoint change — which is what the desktop graph achieved
 * with a Python node that has no browser equivalent.
 */
const FIXUPS = {
  'feature_matching.vn': (graph) => {
    const byId = new Map(graph.nodes.map((n) => [n.id, n]))
    byId.get('src-1').data.params.path = 'samples/car.jpg'
    byId.get('src-2').data.params.path = 'samples/car.jpg'
    byId.get('src-2').data.label = 'Même scène, autre angle'

    graph.nodes.push({
      id: 'rot-1',
      type: 'geom_rotate_no_crop',
      position: { x: 170, y: 300 },
      data: { label: 'Rotation 20°', params: { angle: 20 } },
    })
    for (const edge of graph.edges) {
      if (edge.source === 'src-2') edge.source = 'rot-1'
    }
    graph.edges.push({
      source: 'src-2', sourceHandle: 'image__main',
      target: 'rot-1', targetHandle: 'image__image',
    })
    return graph
  },
}

const availableSamples = new Set(readdirSync(SAMPLES_DIR))

/** Absolute desktop paths become `samples/<file>` when we ship that file. */
function remapPath(original) {
  const filename = String(original).split('/').pop() ?? ''
  if (availableSamples.has(filename)) return `samples/${filename}`

  // The desktop repo keeps full-resolution images; the web build ships 900 px
  // variants under the original name, so a `_900` suffix also resolves.
  const withoutSuffix = filename.replace(/_900(\.[a-z0-9]+)$/i, '$1')
  if (availableSamples.has(withoutSuffix)) return `samples/${withoutSuffix}`

  return filename
}

/** Rewrites one graph in place and reports the paths it still cannot resolve. */
function rewrite(dir, file) {
  const path = join(dir.pathname, file)
  const graph = JSON.parse(readFileSync(path, 'utf8'))
  let rewritten = 0
  const missing = []
  const dropped = []

  // A node wired to nothing that also points at a file we do not ship is a
  // leftover from the desktop session the graph was saved in — it would show up
  // as a broken input in the menu for no reason, so it goes.
  const connected = new Set()
  for (const edge of graph.edges ?? []) {
    connected.add(edge.source)
    connected.add(edge.target)
  }

  for (const node of graph.nodes ?? []) {
    const params = node.data?.params
    if (!params || typeof params.path !== 'string' || params.path === '') continue
    // `path` on an export node names an output folder, not a bundled asset.
    if (node.type === 'util_csv_export' || node.type === 'output_save_frame') continue
    const remapped = remapPath(params.path)
    if (remapped !== params.path) rewritten++
    if (!remapped.startsWith('samples/')) {
      if (!connected.has(node.id)) { dropped.push(node.id); continue }
      missing.push(remapped)
    }
    params.path = remapped
  }

  if (dropped.length > 0) {
    graph.nodes = graph.nodes.filter((node) => !dropped.includes(node.id))
  }
  const fixup = FIXUPS[file]
  if (fixup) fixup(graph)
  writeFileSync(path, JSON.stringify(graph, null, 2))
  return { rewritten, missing, dropped: dropped.length, fixed: Boolean(fixup) }
}

const demoManifest = []
for (const demo of DEMOS) {
  const { rewritten, missing, dropped, fixed } = rewrite(TEMPLATES_DIR, demo.file)
  demoManifest.push({ name: demo.name, description: demo.description, file: demo.file })
  const note = missing.length > 0 ? ` — non fournis : ${missing.join(', ')}` : ''
  const orphan = dropped > 0 ? ` — ${dropped} nœud(s) orphelin(s) retiré(s)` : ''
  console.log(`demo    ${demo.file}: ${rewritten} chemin(s) remappé(s)${note}${orphan}${fixed ? ' — ajusté pour le web' : ''}`)
}
writeFileSync(join(TEMPLATES_DIR.pathname, 'manifest.json'), JSON.stringify(demoManifest, null, 2))

const classManifest = []
for (const group of CLASSES) {
  const entries = []
  for (const entry of group.entries) {
    const { rewritten, missing, dropped } = rewrite(CLASSES_DIR, entry.file)
    entries.push({ name: entry.name, description: entry.description, file: entry.file })
    const note = missing.length > 0 ? ` — non fournis : ${missing.join(', ')}` : ''
    const orphan = dropped > 0 ? ` — ${dropped} nœud(s) orphelin(s) retiré(s)` : ''
    console.log(`class   ${entry.file}: ${rewritten} chemin(s) remappé(s)${note}${orphan}`)
  }
  classManifest.push({ section: group.section, entries })
}
writeFileSync(join(CLASSES_DIR.pathname, 'manifest.json'), JSON.stringify(classManifest, null, 2))

console.log(`\ntemplates/manifest.json : ${demoManifest.length} démos`)
console.log(`classes/manifest.json   : ${classManifest.map((g) => `${g.section} (${g.entries.length})`).join(', ')}`)
