import init, { Machine, SpaceTimeMachine } from './pkg/busy_beaver_blaze.js';

// console.log('[Worker] Starting initialization');
let wasmReady = init();

// self.onmessage = async function (e) {
//     try {
//         // console.log('[Worker] Received message:', e.data);
//         await wasmReady;
//         // console.log('[Worker] WASM initialized');
//         const { programText, goal_x, goal_y, early_stop, x_smoothness, y_smoothness } = e.data;

//         try {
//             const space_time_machine = new SpaceTimeMachine(programText, goal_x, goal_y, x_smoothness, y_smoothness);
//             const CHUNK_SIZE = 10000000n / BigInt(Math.max(x_smoothness, y_smoothness) + 1);
//             let total_steps = 1n;  // Start at 1 since first step is already taken

//             while (true) {
//                 if (early_stop !== null && total_steps >= early_stop) break;

//                 // Calculate next chunk size
//                 let next_chunk = total_steps === 1n ? CHUNK_SIZE - 1n : CHUNK_SIZE;
//                 if (early_stop !== null) {
//                     const remaining = early_stop - total_steps;
//                     next_chunk = remaining < next_chunk ? remaining : next_chunk;
//                 }

//                 // Run the next chunk
//                 const continues = space_time_machine.nth(next_chunk);
//                 total_steps += next_chunk;

//                 // Send intermediate update
//                 const png_data = space_time_machine.png_data();
//                 // console.log('[Worker] Generated PNG data length:', png_data.length);
//                 self.postMessage({
//                     success: true,
//                     intermediate: true,
//                     png_data: png_data,
//                     step_count: space_time_machine.step_count(),
//                     ones_count: space_time_machine.count_ones(),
//                     is_halted: space_time_machine.is_halted()
//                 });

//                 // Exit if machine halted
//                 if (!continues) break;
//             }

//             // Send final result
//             self.postMessage({
//                 success: true,
//                 intermediate: false,
//                 png_data: space_time_machine.png_data(),
//                 step_count: space_time_machine.step_count(),
//                 ones_count: space_time_machine.count_ones(),
//                 is_halted: space_time_machine.is_halted()
//             });
//         } catch (wasmError) {
//             console.error('[Worker] WASM error:', wasmError);
//             throw new Error(wasmError.toString());
//         }
//     } catch (error) {
//         console.error('[Worker] Main error:', error);
//         self.postMessage({
//             success: false,
//             error: error.toString()
//         });
//     }
// };

self.onmessage = async function (e) {
    try {
        await wasmReady;
        const { programText, goal_x, goal_y, early_stop, x_smoothness, y_smoothness } = e.data;

        try {
            const space_time_machine = new SpaceTimeMachine(programText, goal_x, goal_y, x_smoothness, y_smoothness);
            const run_for_seconds = 0.1;

            while (true) {
                if (!space_time_machine.step_for_secs(run_for_seconds, early_stop, 10_000n))
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
