import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { ToneBank, noteFrequencies } from '../../docs/v0.2.8/sound-voices.js';
import { stateColoredProgram } from '../../docs/v0.2.8/state-color.js';
import init, { Machine, SpaceByTimeMachine } from '../../docs/v0.2.8/pkg/busy_beaver_blaze.js';
await init({ module_or_path: await readFile(new URL('../../docs/v0.2.8/pkg/busy_beaver_blaze_bg.wasm', import.meta.url)) });

const rms = values => Math.sqrt(values.reduce((total, value) => total + value * value, 0) / values.length);

test('each state or symbol has a distinct, audible white-key note', () => {
    const frequencies = noteFrequencies(36);
    assert.equal(new Set(frequencies).size, 36);
    assert.ok(frequencies[0] > 200);
    assert.ok(frequencies.at(-1) < 10_000);
    assert.ok(frequencies.every((frequency, index) => index === 0 || frequency > frequencies[index - 1]));
});

test('a retrigger preserves phase and gain, attacks smoothly, then decays exponentially', () => {
    const bank = new ToneBank([250], 48_000, 0.05);
    const silence = new Float32Array(4096);
    bank.render(silence, 0);
    assert.equal(rms(silence), 0);
    const phase = bank.voices[0].phase;
    assert.notEqual(phase, 0); // Oscillator kept running at zero volume.
    bank.schedule(Uint8Array.of(0), 4096, 1);
    const attack = new Float32Array(192);
    bank.render(attack, 4096);
    assert.equal(attack[0], Math.fround(Math.sin(phase) / 192));
    assert.ok(Math.abs(bank.voices[0].gain - 1) < 1e-12);
    bank.render(new Float32Array(2400), 4288);
    assert.ok(Math.abs(bank.voices[0].gain - Math.exp(-1)) < 1e-10);
    const beforeGain = bank.voices[0].gain;
    const beforePhase = bank.voices[0].phase;
    bank.schedule(Uint8Array.of(0), 6688, 1);
    const retrigger = new Float32Array(1);
    bank.render(retrigger, 6688);
    assert.equal(retrigger[0], Math.fround(Math.sin(beforePhase) * (beforeGain + (1 - beforeGain) / 192)));
    assert.ok(bank.voices[0].gain < beforeGain + 0.01);
});

test('overlapping voices are bounded, release cancels queued notes, and long runs compact events', () => {
    const bank = new ToneBank(noteFrequencies(10), 48_000);
    for (let batchIndex = 0; batchIndex < 100; batchIndex++) {
        bank.schedule(Uint8Array.from({ length: 500 }, (_, index) => index % 10), batchIndex * 1200 + 2400, 2.4);
        const output = new Float32Array(1200);
        bank.render(output, batchIndex * 1200);
        assert.ok(output.every(value => Number.isFinite(value) && Math.abs(value) <= 1));
        assert.ok(bank.events.length < 2048);
    }
    bank.schedule(Uint8Array.of(0), 500_000, 1);
    bank.release();
    assert.equal(bank.events.length, 0);
    bank.render(new Float32Array(96_000), 120_000);
    assert.ok(bank.voices.every(voice => voice.gain < 0.00001));
});

function visitedVoices(packet) {
    const voices = new Set();
    for (let index = 1; index < packet.length; index += 3) {
        assert.ok(packet[index] <= packet[0]);
        for (let voice = 0; voice < 36; voice++) {
            if (packet[index + (voice < 32 ? 1 : 2)] & (1 << (voice % 32))) voices.add(voice);
        }
    }
    return [...voices].sort((first, second) => first - second);
}

test('WASM full-speed sound slices report actual source states and read symbols, including halt', () => {
    const states = new SpaceByTimeMachine('1RB1RH_1RC1RH_1RH1RH', 64, 64, true, 0n);
    const symbols = new SpaceByTimeMachine('1RB1RH_1LA0LA', 64, 64, true, 0n);
    assert.deepEqual(visitedVoices(states.step_for_secs_with_sound(0.01, undefined, false)), [0, 1, 2]);
    assert.deepEqual(visitedVoices(symbols.step_for_secs_with_sound(0.01, undefined, true)), [0, 1]);
    assert.equal(states.is_halted(), true);
    assert.equal(symbols.is_halted(), true);
    assert.equal(states.step_for_secs_with_sound(0.01, undefined, false).length, 1);
    states.free(); symbols.free();
});

test('full-speed sound preserves step counts and state-color semantics', () => {
    const program = '1RB0RA_1LC1LF_1RD0LB_1RA1LE_1LI0LC_0RG1LD_1RH1LG_1LC0RG_0LJ0LI_1RZ0LC';
    for (const colored of [false, true]) {
        const text = colored ? stateColoredProgram(program) : program;
        const audible = new SpaceByTimeMachine(text, 64, 64, false, 0n);
        const silent = new SpaceByTimeMachine(text, 64, 64, false, 0n);
        const oracle = new Machine(text);
        for (let batchIndex = 0; batchIndex < 20; batchIndex++) {
            const before = audible.step_count();
            const packet = audible.step_for_secs_with_sound(0.01, before + 127n, false);
            const expected = new Set();
            for (let stepIndex = 0; stepIndex < 127; stepIndex++) {
                expected.add(oracle.state());
                oracle.step();
            }
            assert.deepEqual(visitedVoices(packet), [...expected].sort((first, second) => first - second));
            silent.nth(126n);
            assert.equal(audible.step_count(), silent.step_count());
            assert.equal(audible.count_nonblanks(), silent.count_nonblanks());
            assert.equal(audible.count_nonblanks(), oracle.count_nonblanks());
        }
        assert.deepEqual(audible.to_png(new Uint8Array([255,255,255,0,0,0])), silent.to_png(new Uint8Array([255,255,255,0,0,0])));
        audible.free(); silent.free(); oracle.free();
    }
});

test('audio visit masks above bit 31 retrigger their own voices', () => {
    const bank = new ToneBank(noteFrequencies(36), 48_000);
    bank.scheduleVisits(Uint32Array.of(2000, 2000, 0, 8), 0);
    bank.render(new Float32Array(512), 0);
    assert.ok(bank.voices[35].gain > 0);
    assert.ok(bank.voices.slice(0, 35).every(voice => voice.gain === 0));
});
