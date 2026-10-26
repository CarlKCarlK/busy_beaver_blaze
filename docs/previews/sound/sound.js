import { noteFrequencies } from './sound-voices.js';

export class MachineSound {
    constructor() {
        this.context = null;
        this.node = null;
        this.master = null;
        this.generation = 0;
        this.nextTime = 0;
        this.enabled = false;
        this.ready = false;
        this.volume = 0.2;
        this.decay = 0.15;
    }

    async enable(count, { automatic = false } = {}) {
        clearTimeout(this.idleTimer);
        this.enabled = true;
        this.ready = false;
        this.pendingVisits = [];
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
        // A blocked autoplay resume may stay pending until a later user gesture.
        const resume = this.context.resume();
        const loaded = this.loading.then(() => null, error => error);
        this.master.gain.setTargetAtTime(0, this.context.currentTime, 0.004);
        if (automatic) {
            const resumed = await Promise.race([
                resume.then(() => true),
                new Promise(resolve => setTimeout(() => resolve(false), 150)),
            ]);
            if (!resumed) {
                if (generation === this.generation) this.disable();
                return false;
            }
        }
        const [loadError] = await Promise.all([loaded, resume, new Promise(resolve => setTimeout(resolve, 25))]);
        if (loadError !== null) throw loadError;
        if (!this.enabled || generation !== this.generation) return false;
        this.node.port.postMessage({ type: 'configure', generation, frequencies: noteFrequencies(count), decay: this.decay });
        this.nextTime = this.context.currentTime + 0.05;
        this.ready = true;
        this.master.gain.setTargetAtTime(this.volume, this.context.currentTime, 0.004);
        for (const visits of this.pendingVisits) this.schedule(visits);
        this.pendingVisits = [];
        return true;
    }

    disable() {
        clearTimeout(this.idleTimer);
        this.enabled = false;
        this.ready = false;
        this.pendingVisits = [];
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
        if (!this.enabled || visits.length <= 1) return;
        if (!this.ready) {
            // Retain only a brief startup window while autoplay permission is checked.
            if (this.pendingVisits.length === 4) this.pendingVisits.shift();
            this.pendingVisits.push(visits);
            return;
        }
        const startTime = Math.max(this.nextTime, this.context.currentTime + 0.04);
        this.node.port.postMessage({ type: 'events', generation: this.generation, visits, startTime });
        this.nextTime = startTime + visits[0] / 1_000_000;
    }

    finish() {
        if (!this.context || !this.enabled || !this.ready) return;
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
