// Deterministic audio DSP shared by the worklet and offline tests.
export function noteFrequencies(count) {
    const scale = [0, 2, 4, 5, 7, 9, 11]; // White keys, starting at C4.
    return Array.from({ length: count }, (_, index) => 261.625565 * 2 ** ((scale[index % scale.length] + 12 * Math.floor(index / scale.length)) / 12));
}

export class ToneBank {
    constructor(frequencies, sampleRate, decaySeconds = 0.15) {
        this.sampleRate = sampleRate;
        this.attackSamples = Math.max(1, Math.round(sampleRate * 0.004));
        this.voices = frequencies.map(frequency => ({ phase: 0, increment: 2 * Math.PI * frequency / sampleRate, gain: 0, attackRemaining: 0 }));
        this.events = [];
        this.eventIndex = 0;
        this.setDecay(decaySeconds);
    }

    setDecay(seconds) {
        if (!(seconds > 0)) throw new Error('Decay must be positive.');
        this.decayMultiplier = Math.exp(-1 / (this.sampleRate * seconds));
    }

    schedule(voices, startFrame, framesPerStep) {
        for (let index = 0; index < voices.length; index++) {
            if (voices[index] >= this.voices.length) throw new Error('Sound voice is outside the configured alphabet.');
            this.events.push({ frame: Math.round(startFrame + index * framesPerStep), voice: voices[index] });
        }
    }

    scheduleVisits(slices, startFrame) {
        for (let index = 1; index < slices.length; index += 3) {
            const frame = Math.round(startFrame + slices[index] * this.sampleRate / 1_000_000);
            for (let voice = 0; voice < this.voices.length; voice++) {
                const mask = slices[index + (voice < 32 ? 1 : 2)];
                if (mask & (1 << (voice % 32))) this.events.push({ frame, voice });
            }
        }
    }

    release() {
        this.events = [];
        this.eventIndex = 0;
        for (const voice of this.voices) voice.attackRemaining = 0;
    }

    render(output, firstFrame) {
        const normalization = 1 / Math.max(1, this.voices.length);
        for (let sampleIndex = 0; sampleIndex < output.length; sampleIndex++) {
            const frame = firstFrame + sampleIndex;
            while (this.eventIndex < this.events.length && this.events[this.eventIndex].frame <= frame) {
                const voice = this.voices[this.events[this.eventIndex++].voice];
                // Retrigger the volume attack, never the waveform phase or current gain.
                voice.attackRemaining = this.attackSamples;
            }
            let mixed = 0;
            for (const voice of this.voices) {
                if (voice.attackRemaining > 0) {
                    voice.gain += (1 - voice.gain) / voice.attackRemaining;
                    voice.attackRemaining--;
                } else {
                    voice.gain *= this.decayMultiplier;
                }
                mixed += Math.sin(voice.phase) * voice.gain;
                voice.phase += voice.increment;
                if (voice.phase >= 2 * Math.PI) voice.phase -= 2 * Math.PI;
            }
            output[sampleIndex] = mixed * normalization;
        }
        if (this.eventIndex > 1024 && this.eventIndex < this.events.length) {
            this.events = this.events.slice(this.eventIndex);
            this.eventIndex = 0;
        }
        if (this.eventIndex === this.events.length) {
            this.events.length = 0;
            this.eventIndex = 0;
        }
    }
}
