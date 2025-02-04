import init, { Machine, SpaceTimeMachine } from './pkg/busy_beaver_blaze.js';

let wasmReady = init();

self.onmessage = async function(e) {
    try {
        await wasmReady;
        const { programText, goal_x, goal_y, early_stop } = e.data;
        
        try {
            const space_time_machine = new SpaceTimeMachine(programText, goal_x, goal_y);
            const CHUNK_SIZE = 10000000n;
            let total_steps = 1n;
            
            while (true) {
                // Calculate next chunk size, respecting early_stop if set
                let next_chunk = CHUNK_SIZE;
                if (early_stop !== null) {
                    const remaining = early_stop - total_steps;
                    if (remaining <= 0n) break;
                    next_chunk = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;
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
