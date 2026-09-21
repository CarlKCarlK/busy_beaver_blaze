web_dir := "docs/v0.2.7"

# Build the WebAssembly app and serve it locally.
run:
    wasm-pack build --release --target web --out-dir {{ web_dir }}/pkg
    miniserve docs --index index.html --port 8000 --interfaces 127.0.0.1
