import { stateColoredProgram } from './state-color.js';
import init, { SpaceByTimeMachine } from './pkg/busy_beaver_blaze.js';

// Initialize WASM once
let wasmReady = init();

// Persistent state for live recoloring
let space_by_time_machine = null;              // SpaceByTimeMachine
let currentColors = null;     // Uint8Array of RGB triplets
let stopRequested = false;    // flag to stop stepping but keep machine
let sound = { source: 'off', generation: 0 };
let runCounter = 0;           // increment per start to tag logs

function formatPalette(bytes) {
    if (!bytes || bytes.length < 6 || bytes.length % 3 !== 0) return '(invalid)';
    const toHex = (v) => v.toString(16).padStart(2, '0');
    const out = [];
    for (let i = 0; i < bytes.length / 3; i++) {
        const r = bytes[i * 3], g = bytes[i * 3 + 1], b = bytes[i * 3 + 2];
        out.push(`#${toHex(r)}${toHex(g)}${toHex(b)}`);
    }
    return out.join(', ');
}

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
    if (!space_by_time_machine || !currentColors) return;
    try {
        const png = space_by_time_machine.to_png(currentColors);
        self.postMessage({
            success: true,
            intermediate,
            png_data: png,
            step_count: space_by_time_machine.step_count(),
            ones_count: space_by_time_machine.count_nonblanks(),
            is_halted: space_by_time_machine.is_halted()
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
            const { programText, colorBy, goal_x, goal_y, early_stop, binning, darkMode, colorsBytes } = msg;
            const runId = ++runCounter;

            // Clone the complete palette to avoid aliasing.
            currentColors = (colorsBytes && colorsBytes.length >= 6 && colorsBytes.length % 3 === 0)
                ? new Uint8Array(colorsBytes)
                : defaultPaletteBytes(!!darkMode);

            // Debug: log palette used for this run
            console.log('[worker] run #%d start colors (dark=%s): %s', runId, !!darkMode, formatPalette(currentColors));

            sound = msg.sound || { source: 'off', generation: 0 };
            // Create/replace the machine
            const machine = new SpaceByTimeMachine(colorBy === 'state' ? stateColoredProgram(programText) : programText, goal_x, goal_y, binning, 0n);
            if (space_by_time_machine) space_by_time_machine.free();
            space_by_time_machine = machine;
            stopRequested = false;

            let lastRender = performance.now();
            while (!stopRequested && runId === runCounter) {
                let continues;
                if (sound.source === 'off') {
                    // Preserve the original fast path when sound is disabled.
                    continues = space_by_time_machine.step_for_secs(0.1, early_stop, 10_000n);
                } else {
                    const batchSound = { ...sound };
                    const visits = space_by_time_machine.step_for_secs_with_sound(0.05, early_stop, batchSound.source === 'symbols');
                    // Read symbols refer to the original binary tape, not state-color provenance.
                    if (colorBy === 'state' && batchSound.source === 'symbols') {
                        for (let index = 1; index < visits.length; index += 3) {
                            visits[index + 1] = (visits[index + 1] & 1) | ((visits[index + 1] >>> 1 || visits[index + 2]) ? 2 : 0);
                            visits[index + 2] = 0;
                        }
                    }
                    self.postMessage({ type: 'audio', visits, generation: batchSound.generation }, [visits.buffer]);
                    continues = !space_by_time_machine.is_halted()
                        && (early_stop === undefined || space_by_time_machine.step_count() < early_stop);
                }
                if (!continues) break;
                if (sound.source === 'off' || performance.now() - lastRender >= 100) {
                    await renderAndPost(true);
                    lastRender = performance.now();
                }
                // Yield only to process controls; audio never throttles machine execution.
                await new Promise(resolve => setTimeout(resolve, 0));
            }

            if (runId !== runCounter) return;
            // Final frame
            await renderAndPost(false);
            console.log('[worker] run #%d final frame posted', runId);
            return;
        }

        if (type === 'sound') {
            sound = msg.sound;
            return;
        }

        if (type === 'colors') {
            const { colorsBytes, darkMode } = msg;
            // Update all symbol or writing-state colors, cloning to avoid aliasing.
            currentColors = (colorsBytes && colorsBytes.length >= 6 && colorsBytes.length % 3 === 0)
                ? new Uint8Array(colorsBytes)
                : (currentColors || defaultPaletteBytes(!!darkMode));
            console.log('[worker] update colors (dark=%s): %s', !!darkMode, formatPalette(currentColors));
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
