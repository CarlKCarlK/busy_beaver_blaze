# Busy Beaver Blaze — web v0.2.9

This is the current web release. GitHub Pages serves `main:/docs`; the site root
and the former `/time-scale-preview/` URL redirect here, preserving query
parameters and URL hashes. Earlier version directories remain available.

The runtime is self-contained in this directory: HTML, CSS, state-color
transformation, worker, and generated WASM package. The web version is separate
from the Rust/Python package version.

## Build and preview

```bash
wasm-pack build --release --out-dir docs/v0.2.9/pkg --target web --features image_pyramid
wasm-opt -Oz --strip-debug --strip-dwarf -o docs/v0.2.9/pkg/busy_beaver_blaze_bg.wasm docs/v0.2.9/pkg/busy_beaver_blaze_bg.wasm
python3 -m http.server 8000 --directory docs
```

Open <http://127.0.0.1:8000/v0.2.9/>. Run the WASM regression checks with:

```bash
node --test tests/wasm/space-scale.test.mjs tests/wasm/time-scale.test.mjs tests/wasm/state-color.test.mjs
node tests/wasm/time-scale-browser.mjs
```

The browser check requires Playwright; set `PLAYWRIGHT_MODULE` to its module
path if needed. It checks both entry redirects and verifies that runtime assets
load from v0.2.9. Design and measurement history remain in
[the development notes](../time-scale-preview/README.md).
