import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import init, { PyramidMachine, SpaceByTimeMachine } from '../../docs/time-scale-preview/pkg/busy_beaver_blaze.js';
import referenceInit, { SpaceByTimeMachine as ReferenceMachine } from '../../docs/v0.2.8/pkg/busy_beaver_blaze.js';
import { stateColoredProgram, statePalette } from '../../docs/v0.2.8/state-color.js';

await init({ module_or_path: await readFile(new URL('../../docs/time-scale-preview/pkg/busy_beaver_blaze_bg.wasm', import.meta.url)) });
await referenceInit({ module_or_path: await readFile(new URL('../../docs/v0.2.8/pkg/busy_beaver_blaze_bg.wasm', import.meta.url)) });
const bb5 = '1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RH0LA';
const palette = new Uint8Array([255,255,255,255,165,0]);

test('WASM Quality runs retain history across scale and palette changes, halt and restart', () => {
    for (const program of [bb5, '1RB2LA3RA0LB_3LA2RB1LB0RA', stateColoredProgram(bb5)]) {
        const machine = new PyramidMachine(program, 64, 32, 'linear-b');
        assert.equal(machine.step_count(), 1n);
        while (machine.step_count() < 1001n && !machine.is_halted()) machine.step_for_secs(1, 1001n, 100n);
        const count = machine.step_count();
        const checkpoints = machine.checkpoint_count();
        const memory = machine.retained_bytes();
        const images = [];
        for (const scale of ['linear-b', 'sqrt', 'log']) {
            machine.set_time_scale(scale);
            const png = machine.to_png(statePalette('light', 5));
            assert.deepEqual(Array.from(png.slice(0,8)), [137,80,78,71,13,10,26,10]);
            assert.notDeepEqual(png, machine.to_png(statePalette('dark', 5)));
            images.push(png);
            assert.equal(machine.step_count(), count);
            assert.equal(machine.checkpoint_count(), checkpoints);
            assert.equal(machine.retained_bytes(), memory);
        }
        assert.notDeepEqual(images[0], images[1]);
        assert.notDeepEqual(images[1], images[2]);
        assert.throws(() => machine.set_time_scale('unknown'), /Unknown/);
        machine.free();
    }
    for (const scale of ['linear-b','sqrt','log']) {
        const machine = new PyramidMachine('1RB1LB_1LA1RH', 32, 16, scale);
        assert.equal(machine.step_for_secs(1, 100n, 1n), false);
        assert.equal(machine.is_halted(), true);
        const count = machine.step_count();
        assert.equal(machine.step(), false);
        assert.equal(machine.step_count(), count);
        assert.ok(machine.to_png(palette).length);
        machine.free();
    }
    const fresh = new PyramidMachine(bb5, 64,32,'log');
    assert.equal(fresh.step_count(),1n);
    assert.equal(fresh.checkpoint_count(),1);
    fresh.free();
});

test('Linear A still supports both Quality and Speed', () => {
    for (const binning of [false,true]) {
        const machine = new SpaceByTimeMachine(bb5,64,32,binning,0n);
        machine.step_for_secs(1,1001n,100n);
        const reference = new ReferenceMachine(bb5,64,32,binning,0n);
        reference.step_for_secs(1,1001n,100n);
        assert.deepEqual(machine.to_png(palette), reference.to_png(palette));
        reference.free(); machine.free();
    }
});

test('worker keeps history for scale/color messages and releases replaced machines', async () => {
    const source = await readFile(new URL('../../docs/time-scale-preview/worker.js', import.meta.url),'utf8');
    const messages = [];
    let constructed = 0;
    let freed = 0;
    class TrackedPyramid extends PyramidMachine {
        constructor(...args) { super(...args); constructed++; }
        free() { freed++; super.free(); }
    }
    const self = { postMessage(message) { messages.push(message); } };
    runInNewContext(source.replace(/^import .*;\n/gm,''), {
        self, init: async () => {}, SpaceByTimeMachine, PyramidMachine: TrackedPyramid,
        stateColoredProgram, Uint8Array, setTimeout,
    });
    const send = data => self.onmessage({ data });
    await send({type:'start',programText:bb5,goal_x:64,goal_y:32,early_stop:1001n,binning:true,timeScale:'linear-b',colorsBytes:palette});
    const count = messages.at(-1).step_count;
    for (const timeScale of ['sqrt','log','linear-a','linear-b']) {
        await send({ type:'scale',timeScale });
        assert.equal(messages.at(-1).step_count,count);
        assert.ok(messages.at(-1).png_data.length);
    }
    await send({type:'colors', colorsBytes:new Uint8Array([0,0,0,255,255,255])});
    assert.equal(messages.at(-1).step_count,count);
    assert.equal(constructed,1);
    await send({type:'stop'});
    assert.equal(messages.at(-1).stopped,true);
    await send({type:'start',programText:bb5,goal_x:64,goal_y:32,early_stop:10n,binning:true,timeScale:'linear-b',colorsBytes:palette});
    assert.equal(constructed,2);
    assert.equal(freed,1);
    assert.equal(messages.at(-1).step_count,10n);
    await send({type:'start',programText:bb5,goal_x:64,goal_y:32,early_stop:10n,binning:false,timeScale:'sqrt',colorsBytes:palette});
    assert.equal(messages.at(-1).success,true);
    assert.equal(messages.at(-1).step_count,10n);
    assert.equal(freed,2);
});
