import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import { parseProgram, stateColoredProgram, statePalette } from '../../docs/v0.2.8/state-color.js';
import init, { Machine, SpaceByTimeMachine } from '../../docs/v0.2.8/pkg/busy_beaver_blaze.js';

await init({ module_or_path: await readFile(new URL('../../docs/v0.2.8/pkg/busy_beaver_blaze_bg.wasm', import.meta.url)) });

const champion = '1RB1LB_1LA1RH';

test('all three program formats produce the same colored program', () => {
    const formats = [champion, 'A B\n0 1RB 1LA\n1 1LB 1RH', '0 1\nA 1RB 1LB\nB 1LA 1RH'];
    for (const format of formats) assert.equal(stateColoredProgram(format), '1RB1LB1LB_2LA2RH2RH');
});

test('overwriting one updates provenance, zero clears it, halting writes use their source state', () => {
    const rows = parseProgram(stateColoredProgram('1RB1RH_1LA0LA'));
    // A writes at 0, B writes at 1, then A overwrites the 1 at 0 and halts.
    assert.equal(rows[0][2], '1RH');
    assert.equal(rows[1][1], '0LA');
    assert.equal(rows[1][2], '0LA');
    assert.equal(stateColoredProgram('---1RA'), '---1RA');
});

test('WASM preserves states, head movements, nonblank counts and halt step', () => {
    const programs = [champion, '1RB1RH_1LA0LA', '1RB0LA_1RC1RB_1LA---', '1RB1LB_1LC0RB_1RD1RH_1LA0LC'];
    for (const program of programs) {
        const original = new Machine(program);
        const colored = new Machine(stateColoredProgram(program));
        for (let stepIndex = 0; stepIndex < 1000; stepIndex++) {
            const continues = original.step();
            assert.equal(colored.step(), continues);
            assert.equal(colored.state(), original.state());
            assert.equal(colored.tape_index(), original.tape_index());
            assert.equal(colored.count_nonblanks(), original.count_nonblanks());
            if (!continues) break;
        }
        original.free(); colored.free();
    }
});

test('six-state palettes render through WASM in sampling and binning modes', () => {
    const program = stateColoredProgram('1RB1RH_1RC1RB_1RD1RC_1RE1RD_1RF1RE_1LA1RF');
    for (const binning of [false, true]) {
        const machine = new SpaceByTimeMachine(program, 64, 64, binning, 0n);
        machine.step_for_secs(0.01, 100n, 1n);
        const png = machine.to_png(statePalette('light', 6));
        assert.deepEqual(Array.from(png.slice(0, 8)), [137, 80, 78, 71, 13, 10, 26, 10]);
        machine.free();
    }
    for (const mode of ['light', 'light2', 'dark']) {
        const bytes = statePalette(mode, 6);
        assert.equal(bytes.length, 21);
        assert.equal(new Set(Array.from({ length: 7 }, (_, index) => String(bytes.slice(index * 3, index * 3 + 3)))).size, 7);
    }
});

test('unsupported and malformed programs report actionable errors', () => {
    assert.throws(() => stateColoredProgram('1RA2RA0RA'), /Symbol/);
    assert.throws(() => stateColoredProgram(Array(10).fill('1RA1RA').join('_')), /nine/);
    assert.throws(() => stateColoredProgram('1RA1RA_1RA'), /same number/);
    assert.throws(() => stateColoredProgram('2RA1RA'), /write symbol/);
});


test('every menu preset parses in WASM and binary presets precede multisymbol presets', async () => {
    const page = await readFile(new URL('../../docs/v0.2.8/index.html', import.meta.url), 'utf8');
    const presets = runInNewContext('(' + page.match(/const PROGRAMS = (\{[\s\S]*?\n            \});/)[1] + ')');
    const menu = page.split('<select id="programSelect">')[1].split('</select>')[0];
    let reachedMultisymbol = false;
    for (const [, key, label] of menu.matchAll(/<option value="([^"]+)"[^>]*>([^<]+)<\/option>/g)) {
        if (key === 'custom') continue;
        assert.ok(presets[key], key);
        const rows = parseProgram(presets[key]);
        const machine = new Machine(presets[key]);
        machine.free();
        if (rows[0].length !== 2) reachedMultisymbol = true;
        else {
            assert.equal(reachedMultisymbol, false, key + ' must precede multisymbol presets');
            assert.ok(!label.includes(',2)'), label);
            const colored = new Machine(stateColoredProgram(presets[key]));
            colored.free();
        }
    }
    for (const [key, count] of Object.entries({ binary_counter: 2, bouncer: 3, bell: 3, spaghetti_sandwich: 4, irregular_bell: 4, skelet1: 5, antihydra: 6 })) {
        assert.equal(parseProgram(presets[key]).length, count, key);
    }
});
