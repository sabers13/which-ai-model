# Portfolio page

Source: `templates/index.html.j2`, `assets/page.css`, `assets/page.js`.
The template includes paired English/German copy. Recordings and downloaded
source documents retain their original English and are labelled accordingly.

Build with `make page`. This reads committed evidence and writes a complete static
folder to `dist/which-ai-model/`. It makes no model calls and does not change the
committed results. Preview at the intended subdirectory:

```sh
python3 -m http.server 8317 --directory dist
```

Open `http://localhost:8317/which-ai-model/`. Query parameters support `lang=en|de`,
`theme=light|dark`, `priority=cheapest|most_accurate|fastest` and
`min_score=65|75|85` (default 75). The minimum-score control explores the recorded
measurements; the saved benchmark's three-point rule and recommendations remain
unchanged. When no model reaches the chosen minimum, the row says so.
Language and theme
also persist locally. All priorities and fixed-volume estimates remain readable
without JavaScript. The page also opens directly from disk.
Some Chrome versions block font preloads under `file://`; use the local HTTP
preview for exact typography. Filtering remains functional from disk.

The complete `dist/which-ai-model/` folder is the deployable static site. Keep its
assets, data and downloads together when serving it under `/which-ai-model/`.
Evidence downloads are copied verbatim from their committed sources. Page data
includes source paths for numerical values.

Fonts are self-hosted IBM Plex Sans and Inter with OFL licences in `assets/fonts/`.
Inline Feather icons carry their MIT licence in `assets/icons/`.
