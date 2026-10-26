import { noteFrequencies } from './sound-voices.js';

export class MachineSound {
    constructor() {
        this.context = null;
        this.node = null;
        this.master = null;
        this.generation = 0;
        this.nextTime = 0;
        this.enabled = false;
        this.volume = 0.2;
        this.decay = 0.15;
    }

    async enable(count) {
        clearTimeout(this.idleTimer);
        this.enabled = true;
        const generation = ++this.generation;
        if (!this.context) {
            this.context = new AudioContext();
            this.master = this.context.createGain();
            this.master.gain.value = 0;
            this.master.connect(this.context.destination);
            this.loading = this.context.audioWorklet.addModule(new URL('./sound-worklet.js', import.meta.url)).then(() => {
                this.node = new AudioWorkletNode(this.context, 'machine-sound', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1] });
                this.node.connect(this.master);
            });
        }
        // Resume synchronously from the user's sound selector / Run gesture.
        const resume = this.context.resume();
        this.master.gain.setTargetAtTime(0, this.context.currentTime, 0.004);
        await Promise.all([this.loading, resume, new Promise(resolve => setTimeout(resolve, 25))]);
        if (!this.enabled || generation !== this.generation) return;
        this.node.port.postMessage({ type: 'configure', generation, frequencies: noteFrequencies(count), decay: this.decay });
        this.nextTime = this.context.currentTime + 0.05;
        this.master.gain.setTargetAtTime(this.volume, this.context.currentTime, 0.004);
    }

    disable() {
        clearTimeout(this.idleTimer);
        this.enabled = false;
        if (this.node) this.node.port.postMessage({ type: 'release', generation: this.generation });
        this.generation++;
        if (this.master) {
            this.master.gain.setTargetAtTime(0, this.context.currentTime, 0.004);
            this.idleTimer = setTimeout(() => { if (!this.enabled) this.context.suspend(); }, 40);
        }
    }

    setVolume(volume) {
        this.volume = volume;
        if (this.enabled && this.master) this.master.gain.setTargetAtTime(volume, this.context.currentTime, 0.004);
    }

    setDecay(seconds) {
        this.decay = seconds;
        if (this.node) this.node.port.postMessage({ type: 'decay', seconds });
    }

    schedule(visits) {
        if (!this.enabled || !this.node || visits.length <= 1) return;
        const startTime = Math.max(this.nextTime, this.context.currentTime + 0.04);
        this.node.port.postMessage({ type: 'events', generation: this.generation, visits, startTime });
        this.nextTime = startTime + visits[0] / 1_000_000;
    }

    finish() {
        if (!this.context || !this.enabled) return;
        const generation = this.generation;
        const delay = Math.max(0, this.nextTime - this.context.currentTime) + this.decay * 8;
        this.idleTimer = setTimeout(() => {
            if (generation === this.generation) {
                this.master.gain.setTargetAtTime(0, this.context.currentTime, 0.004);
                this.idleTimer = setTimeout(() => { if (generation === this.generation) this.context.suspend(); }, 40);
            }
        }, delay * 1000);
    }
}
