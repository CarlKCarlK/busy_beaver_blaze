import { ToneBank } from './sound-voices.js';

class MachineSoundProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bank = new ToneBank([], sampleRate);
        this.generation = 0;
        this.port.onmessage = ({ data }) => {
            if (data.type === 'configure') {
                this.generation = data.generation;
                this.bank = new ToneBank(data.frequencies, sampleRate, data.decay);
            } else if (data.type === 'decay') {
                this.bank.setDecay(data.seconds);
            } else if (data.generation === this.generation) {
                if (data.type === 'events') this.bank.scheduleVisits(data.visits, data.startTime * sampleRate);
                else if (data.type === 'release') this.bank.release();
            }
        };
    }

    process(_inputs, outputs) {
        this.bank.render(outputs[0][0], currentFrame);
        return true;
    }
}
registerProcessor('machine-sound', MachineSoundProcessor);
