// Requires a local docs server and Playwright; see the preview README for invocation.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.PREVIEW_URL || 'http://127.0.0.1:8000';
const output = new URL('../../docs/time-scale-preview/comparisons/', import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
    const page = await browser.newPage({ viewport: { width: 1400, height: 1100 } });
    page.setDefaultTimeout(15000);
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    const completed = count => page.waitForFunction(expected =>
        document.querySelector('#stepCount').textContent === expected &&
        document.querySelector('#runButton').textContent === 'Run' &&
        document.querySelector('.image-container img')?.complete, count);
    const image = () => page.locator('.image-container img').getAttribute('src');
    const scale = async value => {
        const previous = await image();
        await page.selectOption('#timeScale', value);
        await page.waitForFunction(previous => document.querySelector('.image-container img')?.src !== previous && document.querySelector('.image-container img')?.complete, previous);
    };

    console.log('Checking reference defaults and halting');
    const space = async value => {
        const previous = await image();
        await page.selectOption('#spaceScale', value);
        await page.waitForFunction(previous => document.querySelector('.image-container img')?.src !== previous && document.querySelector('.image-container img')?.complete, previous);
    };

    // Old URLs default to Linear A. Switching from a reference run asks for a new run.
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb2&run=true`);
    await completed('6');
    assert.equal(await page.locator('#timeScale').inputValue(), 'linear-a');
    assert.equal(await page.locator('#spaceScale').inputValue(), 'linear');
    const oldImage = await image();
    await page.selectOption('#timeScale', 'log');
    assert.match(await page.locator('#timeScaleNotice').textContent(), /Run again/);
    assert.equal(await image(), oldImage);
    assert.equal(await page.locator('#binning').isDisabled(), true);
    await page.click('#runButton');
    await completed('6');
    assert.match(await page.locator('#haltedStatus').textContent(), /Halted/);
    await scale('sqrt');
    await space('log');
    assert.equal(await page.locator('#stepCount').textContent(),'6');
    const beforePalette = await image();
    await page.selectOption('#paletteMode', 'dark');
    await page.waitForFunction(previous => document.querySelector('.image-container img')?.src !== previous, beforePalette);
    assert.equal(await page.locator('#stepCount').textContent(), '6');

    console.log('Checking live controls and stop');
    // Live scale/palette controls, cancellation, then recoloring and rescaling stopped data.
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb6&earlyStop=false&timeScale=sqrt&run=true`);
    await page.waitForSelector('.image-container img');
    await page.selectOption('#timeScale','log');
    await space('sqrt');
    await page.selectOption('#paletteMode','dark');
    await page.click('#runButton');
    await page.waitForFunction(() => document.querySelector('#runButton').textContent === 'Run');
    const stoppedCount = await page.locator('#stepCount').textContent();
    assert.equal(await page.locator('#haltedStatus').textContent(),'Stopped');
    await scale('linear-b');
    assert.equal(await page.locator('#stepCount').textContent(), stoppedCount);
    assert.equal(await page.locator('#haltedStatus').textContent(),'Stopped');
    await page.waitForTimeout(250);
    assert.equal(await page.locator('#stepCount').textContent(), stoppedCount);

    console.log('Checking writing-state colors');
    // Linear A can transform X while retaining Speed and no checkpoints.
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb6&earlyStop=false&binning=false&spaceScale=sqrt&run=true`);
    await page.waitForSelector('.image-container img');
    assert.equal(await page.locator('#binning').isChecked(),false);
    assert.equal(await page.locator('#spaceScale').inputValue(),'sqrt');
    await space('log');
    await page.click('#runButton');
    await page.waitForFunction(() => document.querySelector('#runButton').textContent === 'Run');
    const speedCount = await page.locator('#stepCount').textContent();
    await space('linear');
    assert.equal(await page.locator('#stepCount').textContent(),speedCount);
    assert.match(await page.locator('#pyramidStats').textContent(),/^0 checkpoints/);
    assert.equal(await page.locator('#binning').isChecked(),false);

    // Writing-state colors survive every scale, and a subsequent run starts afresh.
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb5&steps=100001&colorBy=state&timeScale=log&binning=false&run=true`);
    await completed('100,001');
    assert.equal(await page.locator('#binning').isChecked(), true);
    assert.equal(await page.locator('#stateLegend > span').count(), 6);
    await scale('sqrt');
    await space('log');
    await page.selectOption('#paletteMode','dark');
    assert.equal(await page.locator('#stepCount').textContent(),'100,001');

    console.log('Capturing comparisons');
    // Capture all four views of exactly the same machine/history at 100,000 transitions.
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb5&steps=100001&timeScale=linear-b&run=true`);
    await completed('100,001');
    const stats = await page.locator('#pyramidStats').textContent();
    for (const value of ['linear-a','linear-b','sqrt','log']) {
        await scale(value);
        assert.equal(await page.locator('#stepCount').textContent(),'100,001');
        assert.equal(await page.locator('#pyramidStats').textContent(),stats);
        assert.equal(new URL(page.url()).hash.includes(`timeScale=${value}`), value !== 'linear-a');
        await page.screenshot({ path: new URL(`${value}.png`,output).pathname, fullPage: true });
    }
    // Nine distinct mathematical X/Y combinations from the same retained run.
    for (const timeScale of ['linear-b','sqrt','log']) {
        await scale(timeScale);
        for (const spaceScale of ['linear','sqrt','log']) {
            await space(spaceScale);
            assert.equal(await page.locator('#stepCount').textContent(),'100,001');
            assert.equal(await page.locator('#pyramidStats').textContent(),stats);
            assert.equal(new URL(page.url()).hash.includes(`spaceScale=${spaceScale}`),spaceScale!=='linear');
            await page.screenshot({path:new URL(`x-${spaceScale}-y-${timeScale}.png`,output).pathname,fullPage:true});
        }
    }
    assert.deepEqual(errors,[]);
    console.log('Browser checks passed: defaults, restart notice, live scale/palette, stop, halt, state colors, hash, four time screenshots and nine X/Y screenshots.');
} finally {
    await browser.close();
}
