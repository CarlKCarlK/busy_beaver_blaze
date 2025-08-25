import init, { Machine, SpaceByTimeMachine } from './pkg/busy_beaver_blaze.js';

// console.log('[Worker] Starting initialization');
let wasmReady = init();

self.onmessage = async function (e) {
    try {
        await wasmReady;
        const { programText, goal_x, goal_y, early_stop, binning, darkMode } = e.data;

        try {
            const space_time_machine = new SpaceByTimeMachine(programText, goal_x, goal_y, binning, 0n);
            const run_for_seconds = 0.1;

            // Set colors based on dark mode
            const colors = darkMode
                ? new Uint8Array([
                    // black
                    0, 0, 0,
                    // white
                    255, 255, 255,
                    // 50% grey
                    128, 128, 128,
                    // light grey
                    192, 192, 192,
                    // dark gray
                    64, 64, 64
                ])
                : new Uint8Array();


            while (true) {
                if (!space_time_machine.step_for_secs(run_for_seconds, early_stop, 10_000n))
                    break;

                self.postMessage({
                    success: true,
                    intermediate: true,
                    png_data: space_time_machine.to_png(colors),
                    step_count: space_time_machine.step_count(),
                    ones_count: space_time_machine.count_nonblanks(),
                    is_halted: space_time_machine.is_halted()
                });

            }

            // Send final result
            self.postMessage({
                success: true,
                intermediate: false,
                png_data: space_time_machine.to_png(colors),
                step_count: space_time_machine.step_count(),
                ones_count: space_time_machine.count_nonblanks(),
                is_halted: space_time_machine.is_halted()
            });
        } catch (wasmError) {
            throw new Error(wasmError.toString());
        }
    } catch (error) {
        self.postMessage({
            success: false,
            error: error.toString()
        });
    }
};
