import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import init, { PyramidMachine, SpaceByTimeMachine } from '../../docs/v0.2.9/pkg/busy_beaver_blaze.js';
import referenceInit, { SpaceByTimeMachine as ReferenceMachine } from '../../docs/v0.2.8/pkg/busy_beaver_blaze.js';
import { stateColoredProgram, statePalette } from '../../docs/v0.2.9/state-color.js';
await init({ module_or_path: await readFile(new URL('../../docs/v0.2.9/pkg/busy_beaver_blaze_bg.wasm',import.meta.url)) });
await referenceInit({ module_or_path: await readFile(new URL('../../docs/v0.2.8/pkg/busy_beaver_blaze_bg.wasm',import.meta.url)) });
const bb5 = '1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RH0LA';
const palette = statePalette('light',5);
const programs = [bb5,'1RB2LA3RA0LB_3LA2RB1LB0RA',stateColoredProgram(bb5)];

test('Linear A identity and all X maps preserve Quality/Speed, time count and palette support', () => {
    for (const program of programs) for (const binning of [false,true]) {
        const machine = new SpaceByTimeMachine(program,16,9,binning,0n);
        const reference = new ReferenceMachine(program,16,9,binning,0n);
        machine.step_for_secs(1,1001n,1n);
        reference.step_for_secs(1,1001n,1n);
        assert.deepEqual(machine.to_spatial_png(palette,'linear'),reference.to_png(palette));
        const count = machine.step_count();
        const referencePng = machine.to_png(palette);
        for (const spaceScale of ['row','linear']) {
            const png = machine.to_spatial_png(palette,spaceScale);
            assert.equal(machine.step_count(),count);
            assert.equal(png[0],137);
            assert.deepEqual(png.slice(16,24),referencePng.slice(16,24)); // Same PNG dimensions.
            if (spaceScale !== 'linear') assert.notDeepEqual(png,referencePng);
            assert.notDeepEqual(png,machine.to_spatial_png(statePalette('dark',5),spaceScale));
        }
        assert.throws(() => machine.to_spatial_png(palette,'perspective'), /Unknown space scale/);
        assert.equal(machine.checkpoint_count,undefined);
        machine.free(); reference.free();
    }
});

test('all six pyramid X/Y combinations reuse history and symbol/state colors', () => {
    for (const sampling of [false,true]) for (const program of programs) {
        const machine = sampling ? PyramidMachine.sampling(program,32,16,'linear-b') : new PyramidMachine(program,32,16,'linear-b');
        machine.step_for_secs(1,1001n,1n);
        const count = machine.step_count();
        const checkpoints = machine.checkpoint_count();
        const bytes = machine.retained_bytes();
        for (const timeScale of ['linear-b','sqrt','log']) for (const spaceScale of ['linear','row']) {
            machine.set_time_scale(timeScale);
            machine.set_space_scale(spaceScale);
            const png = machine.to_png(palette);
            assert.equal(png[0],137);
            assert.equal(machine.step_count(),count);
            assert.equal(machine.checkpoint_count(),checkpoints);
            assert.equal(machine.retained_bytes(),bytes);
            assert.notDeepEqual(png,machine.to_png(statePalette('dark',5)));
        }
        for (const removed of ['sqrt','log','square','exp','perspective']) assert.throws(() => machine.set_space_scale(removed), /Unknown space scale/);
        machine.free();
    }
});

test('worker handles all eight combinations and does not construct a pyramid just for X', async () => {
    const source = await readFile(new URL('../../docs/v0.2.9/worker.js',import.meta.url),'utf8');
    let constructed = 0;
    class TrackedPyramid extends PyramidMachine {
        constructor(...args) { super(...args); constructed++; }
    }
    const messages = [];
    const self = { postMessage(message) { messages.push(message); } };
    runInNewContext(source.replace(/^import .*;\n/gm,''),{ self, init:async()=>{},SpaceByTimeMachine,PyramidMachine:TrackedPyramid,stateColoredProgram,Uint8Array,setTimeout });
    const send = data => self.onmessage({ data });
    for (const binning of [true,false]) {
        await send({type:'start',programText:bb5,goal_x:32,goal_y:16,early_stop:1001n,binning,timeScale:'linear-a',spaceScale:'row',colorsBytes:palette});
        assert.equal(constructed,0);
        const count = messages.at(-1).step_count;
        for (const spaceScale of ['row','linear']) {
            await send({type:'space',spaceScale});
            assert.equal(messages.at(-1).step_count,count);
            assert.equal(messages.at(-1).checkpoints,0);
            assert.ok(messages.at(-1).png_data.length);
        }
    }
    await send({type:'start',programText:bb5,goal_x:32,goal_y:16,early_stop:1001n,binning:true,timeScale:'linear-b',spaceScale:'row',colorsBytes:palette});
    const count = messages.at(-1).step_count;
    const checkpoints = messages.at(-1).checkpoints;
    for (const timeScale of ['linear-a','linear-b','sqrt','log']) for (const spaceScale of ['linear','row']) {
        await send({type:'scale',timeScale});
        await send({type:'space',spaceScale});
        assert.equal(messages.at(-1).step_count,count);
        assert.equal(messages.at(-1).checkpoints,checkpoints);
        assert.ok(messages.at(-1).png_data.length);
    }
    assert.equal(constructed,1);
    await send({type:'stop'});
    assert.equal(messages.at(-1).stopped,true);
    await send({type:'space',spaceScale:'row'});
    assert.equal(messages.at(-1).step_count,count);
    await send({type:'start',programText:'1RB1LB_1LA1RH',goal_x:32,goal_y:16,early_stop:1001n,binning:true,timeScale:'linear-b',spaceScale:'row',colorsBytes:palette});
    assert.equal(messages.at(-1).is_halted,true);
    const haltedCount = messages.at(-1).step_count;
    await send({type:'space',spaceScale:'row'});
    assert.equal(messages.at(-1).step_count,haltedCount);
});
