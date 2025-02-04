import init, { Machine } from './pkg/busy_beaver_blaze.js';

let wasmReady = init();

self.onmessage = async function(e) {
    try {
        await wasmReady;
        const { programText, goal_x, goal_y, early_stop } = e.data;
        
        // Capture any WASM errors
        try {
            const machine = new Machine(programText);
            const result = machine.space_time(goal_x, goal_y, early_stop);  // Changed from space_time_js to space_time
            
            self.postMessage({
                success: true,
                png_data: result.png_data(),
                step_count: result.step_count(),
                ones_count: machine.count_ones(),
                is_halted: machine.is_halted()
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
