// Reproducible before/after WASM comparison against the pushed time-only checkpoint.
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';
import init, { SpaceByTimeMachine as After } from '../../docs/time-scale-preview/pkg/busy_beaver_blaze.js';
const temporary = await mkdtemp(join(tmpdir(),'space-scale-baseline-'));
try {
    for (const artifact of ['busy_beaver_blaze.js','busy_beaver_blaze_bg.wasm','package.json']) {
        await writeFile(join(temporary,artifact),execFileSync('git',['show',`bb97716:docs/time-scale-preview/pkg/${artifact}`]));
    }
    const {default:beforeInit,SpaceByTimeMachine:Before} = await import(pathToFileURL(join(temporary,'busy_beaver_blaze.js')));
    await beforeInit({module_or_path:await readFile(join(temporary,'busy_beaver_blaze_bg.wasm'))});
    await init({module_or_path:await readFile(new URL('../../docs/time-scale-preview/pkg/busy_beaver_blaze_bg.wasm',import.meta.url))});
    const results = {};
    for (let trial=0;trial<10;trial++) for (const binning of [true,false]) {
        for (const [name,Class] of [['before',Before],['after',After]]) {
            const machine = new Class('1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RH0LA',128,128,binning,0n);
            const start = performance.now();
            for(let frame=0;frame<10;frame++) {
                machine.nth(99999n);
                machine.to_png(new Uint8Array([255,255,255,255,165,0]));
            }
            (results[`${name} ${binning}`] ||= []).push(performance.now()-start);
            machine.free();
        }
    }
    for(const [name,values] of Object.entries(results)) {
        const steady = values.slice(2).sort((first,second)=>first-second);
        console.log(`${name}: median ${(0.5*(steady[3]+steady[4])).toFixed(3)} ms`);
    }
    await writeFile(new URL('../../docs/time-scale-preview/results/space/wasm-timings.json',import.meta.url),JSON.stringify(results,null,2)+'\n');
} finally {
    await rm(temporary,{recursive:true,force:true});
}
