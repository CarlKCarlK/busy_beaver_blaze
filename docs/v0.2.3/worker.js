import init, { Machine, SpaceTimeMachine } from './pkg/busy_beaver_blaze.js';

let wasmReady = init();

self.onmessage = async function (e) {
    try {
        await wasmReady;
        const { programText, goal_x, goal_y, early_stop, x_smoothness, y_smoothness } = e.data;

        try {
            const space_time_machine = new SpaceTimeMachine(programText, goal_x, goal_y, x_smoothness, y_smoothness);
            const run_for_seconds = 1.0;

            while (true) {
                if (!space_time_machine.step_for_secs(run_for_seconds, early_stop))
                    break;

                self.postMessage({
                    success: true,
                    intermediate: true,
                    png_data: space_time_machine.png_data(),
                    step_count: space_time_machine.step_count(),
                    ones_count: space_time_machine.count_ones(),
                    is_halted: space_time_machine.is_halted()
                });

            }

            // Send final result
            self.postMessage({
                success: true,
                intermediate: false,
                png_data: space_time_machine.png_data(),
                step_count: space_time_machine.step_count(),
                ones_count: space_time_machine.count_ones(),
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
