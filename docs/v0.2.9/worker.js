import { stateColoredProgram } from './state-color.js';
import init, { SpaceByTimeMachine, PyramidMachine } from './pkg/busy_beaver_blaze.js';

const wasmReady = init();
let machine = null;
let pyramid = false;
let currentColors = null;
let scale = 'linear-a';
let spaceScale = 'linear';
let runCounter = 0;
let running = false;

function renderAndPost(intermediate) {
    if (!machine || !currentColors) return;
    const png = pyramid
        ? (scale === 'linear-a' ? machine.linear_png(currentColors) : machine.to_png(currentColors))
        : (spaceScale === 'linear' ? machine.to_png(currentColors) : machine.to_spatial_png(currentColors, spaceScale));
    self.postMessage({
        success: true, intermediate, png_data: png,
        step_count: machine.step_count(), ones_count: machine.count_nonblanks(),
        is_halted: machine.is_halted(),
        checkpoints: pyramid ? machine.checkpoint_count() : 0,
        pyramid_bytes: pyramid ? machine.retained_bytes() : 0,
    });
}

self.onmessage = async ({ data: msg }) => {
    await wasmReady;
    try {
        if (msg.type === 'start') {
            const runId = ++runCounter;
            running = false;
            if (machine) machine.free();
            machine = null;
            scale = msg.timeScale || 'linear-a';
            spaceScale = msg.spaceScale || 'linear';
            pyramid = scale !== 'linear-a';
            const program = msg.colorBy === 'state' ? stateColoredProgram(msg.programText) : msg.programText;
            currentColors = new Uint8Array(msg.colorsBytes);
            machine = pyramid
                ? (msg.binning ? new PyramidMachine(program, msg.goal_x, msg.goal_y, scale)
                    : PyramidMachine.sampling(program, msg.goal_x, msg.goal_y, scale))
                : new SpaceByTimeMachine(program, msg.goal_x, msg.goal_y, msg.binning, 0n);
            if (pyramid) machine.set_space_scale(spaceScale);
            running = true;
            while (running && runId === runCounter) {
                const continuing = machine.step_for_secs(0.1, msg.early_stop, 10_000n);
                if (!continuing) {
                    running = false;
                    renderAndPost(false);
                    return;
                }
                renderAndPost(true);
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            return;
        }
        if (msg.type === 'stop') {
            running = false;
            renderAndPost(true);
            self.postMessage({ success: true, intermediate: false, stopped: true });
            return;
        }
        if (msg.type === 'colors') {
            currentColors = new Uint8Array(msg.colorsBytes);
            renderAndPost(true);
            return;
        }
        if (msg.type === 'space') {
            spaceScale = msg.spaceScale || 'linear';
            if (pyramid && machine) machine.set_space_scale(spaceScale);
            renderAndPost(true);
            return;
        }
        if (msg.type === 'scale') {
            if (!machine) return;
            const nextScale = msg.timeScale;
            if (!['linear-a','linear-b','sqrt','log'].includes(nextScale)) throw new Error(`Unknown time scale: ${nextScale}`);
            if (!pyramid && nextScale !== 'linear-a') {
                machine = PyramidMachine.from_linear(machine);
                pyramid = true;
                machine.set_space_scale(spaceScale);
            }
            if (pyramid && nextScale !== 'linear-a') machine.set_time_scale(nextScale);
            scale = nextScale;
            renderAndPost(true);
            return;
        }
        throw new Error(`Unknown message type: ${msg.type}`);
    } catch (error) {
        running = false;
        self.postMessage({ success: false, error: String(error) });
    }
};
