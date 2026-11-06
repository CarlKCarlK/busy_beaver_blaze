// Compare shipped Linear A WASM with Linear A in the opt-in preview build.
import { readFile, writeFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import beforeInit, { SpaceByTimeMachine as Before } from '../../docs/v0.2.8/pkg/busy_beaver_blaze.js';
import afterInit, { SpaceByTimeMachine as After } from '../../docs/v0.2.9/pkg/busy_beaver_blaze.js';
await beforeInit({ module_or_path: await readFile(new URL('../../docs/v0.2.8/pkg/busy_beaver_blaze_bg.wasm',import.meta.url)) });
await afterInit({ module_or_path: await readFile(new URL('../../docs/v0.2.9/pkg/busy_beaver_blaze_bg.wasm',import.meta.url)) });
const program = '1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RH0LA';
const results = {};
for (let trial = 0; trial < 10; trial++) {
    for (const binning of [true,false]) {
        for (const [name,Class] of [['before',Before],['after',After]]) {
            const machine = new Class(program,128,128,binning,0n);
            const start = performance.now();
            for (let frame = 0; frame < 10; frame++) {
                machine.nth(99999n);
                machine.to_png(new Uint8Array([255,255,255,255,165,0]));
            }
            (results[`${name} ${binning}`] ||= []).push(performance.now() - start);
            machine.free();
        }
    }
}
for (const [name,values] of Object.entries(results)) {
    const steady = values.slice(2).sort((first,second) => first - second);
    console.log(`${name}: median ${(0.5 * (steady[3] + steady[4])).toFixed(3)} ms`);
}
await writeFile(new URL('../../docs/time-scale-preview/results/wasm-timings.json',import.meta.url), JSON.stringify(results,null,2) + '\n');
