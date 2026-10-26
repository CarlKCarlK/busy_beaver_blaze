import assert from 'node:assert/strict';
import test from 'node:test';
import { MachineSound } from '../../docs/v0.2.8/sound.js';

function mockBrowser(allowAutoplay) {
    const previousContext = globalThis.AudioContext;
    const previousNode = globalThis.AudioWorkletNode;
    globalThis.AudioContext = class {
        currentTime = 0;
        state = 'suspended';
        destination = {};
        audioWorklet = { addModule: async () => {} };
        createGain() { return { gain: { value: 0, setTargetAtTime() {} }, connect() {} }; }
        resume() {
            if (!allowAutoplay) return new Promise(() => {});
            this.state = 'running';
            return Promise.resolve();
        }
        suspend() { this.state = 'suspended'; return Promise.resolve(); }
    };
    globalThis.AudioWorkletNode = class {
        messages = [];
        port = { postMessage: message => this.messages.push(message) };
        connect() {}
    };
    return () => { globalThis.AudioContext = previousContext; globalThis.AudioWorkletNode = previousNode; };
}

test('blocked autoplay returns without waiting for a user gesture or retaining pending sound', async () => {
    const restore = mockBrowser(false);
    const sound = new MachineSound();
    try {
        const enabling = sound.enable(2, { automatic: true });
        sound.schedule(Uint32Array.of(2000, 2000, 1, 0));
        assert.equal(await enabling, false);
        assert.equal(sound.enabled, false);
        assert.equal(sound.ready, false);
        assert.equal(sound.pendingVisits.length, 0);
    } finally { sound.disable(); restore(); }
});

test('allowed autoplay retains a short machine’s sound while audio initializes', async () => {
    const restore = mockBrowser(true);
    const sound = new MachineSound();
    try {
        const enabling = sound.enable(2, { automatic: true });
        sound.schedule(Uint32Array.of(2000, 2000, 3, 0));
        sound.finish(); // The visualization may finish before audio is ready.
        assert.equal(sound.idleTimer, undefined);
        assert.equal(await enabling, true);
        assert.equal(sound.ready, true);
        assert.equal(sound.context.state, 'running');
        assert.deepEqual(sound.node.messages.map(message => message.type), ['configure', 'events']);
        assert.ok(sound.nextTime > 0.05);
        assert.equal(sound.pendingVisits.length, 0);
    } finally { sound.disable(); restore(); }
});
