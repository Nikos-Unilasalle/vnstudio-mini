# vnstudio-mini

Version web légère de VNStudio pour les TD de Vision par ordinateur (série 1). 100% statique — tout tourne dans le navigateur via OpenCV.js (WASM), aucun serveur.

## Dev

```bash
npm install
npm run dev
```

## Build / déploiement GitHub Pages

Le workflow `.github/workflows/deploy.yml` build et déploie automatiquement sur push vers `main`, à condition d'activer GitHub Pages → Source: GitHub Actions dans les paramètres du repo.

Si le repo n'est pas nommé `vnstudio-mini`, adapte `base` dans [vite.config.ts](vite.config.ts) (ou passe `VITE_BASE=/mon-repo/` au build).

## Scope actuel

- Skeleton complet : canvas ReactFlow, palette de nodes, panneau de paramètres, panneau de preview, import/export `.vn`.
- **TD I (galets)** entièrement fonctionnel : Image File, Grayscale, Mask Polygon, Threshold (Advanced), Morphology (Advanced), Fill Holes, Distance Transform, Connected Components, Connected Comp. (CV2), Marker Filter, Watershed, Region Filter, Region Properties, Visual Calibration, Grain Size Histogram, Display, Inspector, CSV Export.
- Le bouton "Charger TD I" importe directement `docs/Missions/Serie 1/M1.1_reference.vn` du repo VNStudio principal (copié dans `public/samples/`).
- TD II (empreintes) et TD III (gaze/MediaPipe) restent à porter sur cette même base — mêmes conventions de node (`src/engine/types.ts`, `src/nodes/*.ts`, enregistrement dans `src/engine/registry.ts`).

## Ajouter un node

1. Créer `src/nodes/monNode.ts` exportant un `NodeDef` (voir `src/nodes/grayscale.ts` pour un exemple minimal).
2. L'enregistrer dans `src/engine/registry.ts`.
3. Les ports colorés suivent la convention VNStudio : image=bleu, mask=gris, scalar=jaune, regions=vert, dict=violet, points=rose, string=orange (`src/ui/portColors.ts`).
