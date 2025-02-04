import init, { Machine, SpaceTimeMachine } from './pkg/busy_beaver_blaze.js';

let wasmReady = init();

self.onmessage = async function(e) {
    try {
        await wasmReady;
        const { programText, goal_x, goal_y, early_stop } = e.data;
        
        // Capture any WASM errors
        try {
            const space_time_machine = new SpaceTimeMachine(programText, goal_x, goal_y);

            space_time_machine.nth(early_stop);
            
            self.postMessage({
                success: true,
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
