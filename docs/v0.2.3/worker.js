import init, { Machine, SpaceTimeMachine } from './pkg/busy_beaver_blaze.js';

let wasmReady = init();

self.onmessage = async function (e) {
    try {
        await wasmReady;
        const { programText, goal_x, goal_y, early_stop } = e.data;

        try {
            const space_time_machine = new SpaceTimeMachine(programText, goal_x, goal_y);
            const CHUNK_SIZE = 10000000n;
            let total_steps = 1n;  // Start at 1 since first step is already taken

            while (true) {
                if (early_stop !== null && total_steps >= early_stop) break;

                // Calculate next chunk size
                let next_chunk = total_steps === 1n ? CHUNK_SIZE - 1n : CHUNK_SIZE;
                if (early_stop !== null) {
                    const remaining = early_stop - total_steps;
                    next_chunk = remaining < next_chunk ? remaining : next_chunk;
                }

                // Run the next chunk
                const continues = space_time_machine.nth(next_chunk);
                total_steps += next_chunk;

                // Send intermediate update
                self.postMessage({
                    success: true,
                    intermediate: true,
                    png_data: space_time_machine.png_data(),
                    step_count: space_time_machine.step_count(),
                    ones_count: space_time_machine.count_ones(),
                    is_halted: space_time_machine.is_halted()
                });

                // Exit if machine halted
                if (!continues) break;
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
