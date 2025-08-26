import init, { Machine, SpaceByTimeMachine } from './pkg/busy_beaver_blaze.js';

// Initialize WASM once
let wasmReady = init();

// Persistent state for live recoloring
let sbtm = null;              // SpaceByTimeMachine
let currentColors = null;     // Uint8Array(15) - always 5 colors
let stopRequested = false;    // flag to stop stepping but keep machine

function defaultPaletteBytes(isDark) {
    if (isDark) {
        return new Uint8Array([
            255, 255, 255, // white (symbol 0)
            0, 0, 0,       // black (symbol 1)
            128, 128, 128, // 50% gray (symbol 2)
            64, 64, 64,    // 25% gray (symbol 3)
            192, 192, 192  // 75% gray (symbol 4)
        ]);
    }
    return new Uint8Array([
        255, 255, 255, // white (symbol 0)
        255, 165, 0,   // orange (symbol 1)
        255, 255, 0,   // yellow (symbol 2)
        255, 0, 255,   // magenta (symbol 3)
        0, 255, 255    // cyan (symbol 4)
    ]);
}

async function renderAndPost(intermediate) {
    if (!sbtm || !currentColors) return;
    try {
        const png = sbtm.to_png(currentColors);
        self.postMessage({
            success: true,
            intermediate,
            png_data: png,
            step_count: sbtm.step_count(),
            ones_count: sbtm.count_nonblanks(),
            is_halted: sbtm.is_halted()
        });
    } catch (e) {
        self.postMessage({ success: false, error: e.toString() });
    }
}

self.onmessage = async function (e) {
    await wasmReady;
    const msg = e.data || {};
    const type = msg.type || 'start';

    try {
        if (type === 'start') {
            const { programText, goal_x, goal_y, early_stop, binning, darkMode, colorsBytes } = msg;

            // Normalize to fixed 5-color palette (15 bytes)
            currentColors = (colorsBytes && colorsBytes.length === 15)
                ? colorsBytes
                : defaultPaletteBytes(!!darkMode);

            // Create/replace the machine
            sbtm = new SpaceByTimeMachine(programText, goal_x, goal_y, binning, 0n);
            stopRequested = false;

            const run_for_seconds = 0.1;
            while (true) {
                if (stopRequested || !sbtm.step_for_secs(run_for_seconds, early_stop, 10_000n)) {
                    break;
                }
                await renderAndPost(true);
                // Yield to event loop so we can process incoming 'colors' messages
                await new Promise((resolve) => setTimeout(resolve, 0));
            }

            // Final frame
            await renderAndPost(false);
            return;
        }

        if (type === 'colors') {
            const { colorsBytes, darkMode } = msg;
            // Update colors (keep 5-color size)
            currentColors = (colorsBytes && colorsBytes.length === 15)
                ? colorsBytes
                : (currentColors || defaultPaletteBytes(!!darkMode));
            // Re-render instantly
            await renderAndPost(true);
            return;
        }

        if (type === 'stop') {
            // Stop stepping but keep machine for recoloring
            stopRequested = true;
            self.postMessage({ success: true, intermediate: false, stopped: true });
            return;
        }

        self.postMessage({ success: false, error: `Unknown message type: ${type}` });
    } catch (err) {
        self.postMessage({ success: false, error: err.toString() });
    }
};
