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
    assert.deepEqual(await page.locator('#spaceScale option').evaluateAll(options => options.map(option=>option.value)),['linear','row']);
    assert.deepEqual(await page.locator('#timeScale option').evaluateAll(options => options.map(option=>option.value)),['linear-a','sqrt']);
    assert.deepEqual(await page.locator('#timeScale option').allTextContents(),['Linear','Square Root']);
    const creditButton = page.getByRole('button',{name:'About square-root time scaling'});
    await creditButton.focus();
    assert.equal(await page.locator('#timeScaleCredit').isVisible(),true);
    assert.equal(await page.locator('#timeScaleCredit a').getAttribute('href'),'https://github.com/opfromthestart');
    assert.match(await page.locator('#timeScaleCredit').textContent(),/triangular-number sampling/);
    await page.locator('#timeScale').focus();
    assert.equal(await page.locator('#timeScaleCredit').isVisible(),false);
    await creditButton.hover();
    assert.equal(await page.locator('#timeScaleCredit').isVisible(),true);
    await page.locator('#programInput').hover();
    const gap = await page.evaluate(() => document.getElementById('timeScale').closest('.controls').getBoundingClientRect().top - document.getElementById('programInput').getBoundingClientRect().bottom);
    assert.ok(gap <= 16, `Program/control gap too large: ${gap}`);
    const oldImage = await image();
    await page.selectOption('#timeScale', 'sqrt');
    assert.match(await page.locator('#timeScaleNotice').textContent(), /Run again/);
    assert.equal(await image(), oldImage);
    assert.equal(await page.locator('#binning').isDisabled(), false);
    await page.click('#runButton');
    await completed('6');
    assert.match(await page.locator('#haltedStatus').textContent(), /Halted/);
    await scale('sqrt');
    await space('row');
    assert.equal(await page.locator('#stepCount').textContent(),'6');
    const beforePalette = await image();
    await page.selectOption('#paletteMode', 'dark');
    await page.waitForFunction(previous => document.querySelector('.image-container img')?.src !== previous, beforePalette);
    assert.equal(await page.locator('#stepCount').textContent(), '6');

    console.log('Checking live controls and stop');
    // Live scale/palette controls, cancellation, then recoloring and rescaling stopped data.
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb6&earlyStop=false&timeScale=sqrt&run=true`);
    await page.waitForSelector('.image-container img');
    await page.selectOption('#timeScale','linear-a');
    await space('row');
    await page.selectOption('#paletteMode','dark');
    await page.click('#runButton');
    await page.waitForFunction(() => document.querySelector('#runButton').textContent === 'Run');
    const stoppedCount = await page.locator('#stepCount').textContent();
    assert.equal(await page.locator('#haltedStatus').textContent(),'Stopped');
    await scale('sqrt');
    assert.equal(await page.locator('#stepCount').textContent(), stoppedCount);
    assert.equal(await page.locator('#haltedStatus').textContent(),'Stopped');
    await page.waitForTimeout(250);
    assert.equal(await page.locator('#stepCount').textContent(), stoppedCount);

    console.log('Checking writing-state colors');
    // Linear A can transform X while retaining Speed and no checkpoints.
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb6&earlyStop=false&binning=false&spaceScale=row&run=true`);
    await page.waitForSelector('.image-container img');
    assert.equal(await page.locator('#binning').isChecked(),false);
    assert.equal(await page.locator('#spaceScale').inputValue(),'row');
    await space('row');
    await page.click('#runButton');
    await page.waitForFunction(() => document.querySelector('#runButton').textContent === 'Run');
    const speedCount = await page.locator('#stepCount').textContent();
    await space('linear');
    assert.equal(await page.locator('#stepCount').textContent(),speedCount);
    assert.equal(await page.locator('#binning').isChecked(),false);

    // Writing-state colors survive every scale, and a subsequent run starts afresh.
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb5&steps=100001&colorBy=state&timeScale=sqrt&binning=false&run=true`);
    await completed('100,001');
    assert.equal(await page.locator('#binning').isChecked(), false);
    assert.equal(await page.locator('#stateLegend > span').count(), 6);
    await scale('sqrt');
    await space('row');
    await page.selectOption('#paletteMode','dark');
    assert.equal(await page.locator('#stepCount').textContent(),'100,001');

    console.log('Capturing comparisons');
    // Capture both time views of exactly the same machine/history at 100,000 transitions.
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb5&steps=100001&timeScale=sqrt&run=true`);
    await completed('100,001');
    assert.equal(await page.locator('#pyramidStats').count(),0);
    for (const value of ['linear-a','sqrt']) {
        await scale(value);
        assert.equal(await page.locator('#stepCount').textContent(),'100,001');
        assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(1)).get('timeScale'),value === 'linear-a' ? 'linear' : value);
        await page.screenshot({ path: new URL(`${value}.png`,output).pathname, fullPage: true });
    }
    // Both X modes with both time modes, reusing the same history.
    for (const timeScale of ['linear-a','sqrt']) {
        await scale(timeScale);
        for (const spaceScale of ['linear','row']) {
            await space(spaceScale);
            assert.equal(await page.locator('#stepCount').textContent(),'100,001');
            assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(1)).get('spaceScale'),spaceScale);
            if (timeScale === 'sqrt') await page.screenshot({path:new URL(`x-${spaceScale}-y-${timeScale}.png`,output).pathname,fullPage:true});
        }
    }
    await scale('sqrt');
    await space('row');
    assert.equal(await page.locator('#stepCount').textContent(),'100,001');
    await page.screenshot({path:new URL('x-row-y-sqrt.png',output).pathname,fullPage:true});
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb5&steps=1001&timeScale=sqrt&spaceScale=row&binning=false&run=true`);
    await completed('1,001');
    assert.equal(await page.locator('#spaceScale').inputValue(),'row');
    assert.equal(await page.locator('#binning').isChecked(),false);
    for (const [timeValue,spaceValue] of [['linear','linear'],['sqrt','row']]) {
        const previous = await image();
        await page.evaluate(([timeValue,spaceValue]) => {location.hash = `program=bb5&steps=1001&timeScale=${timeValue}&spaceScale=${spaceValue}&binning=false`;},[timeValue,spaceValue]);
        await page.waitForFunction(previous => document.querySelector('.image-container img')?.src !== previous && document.querySelector('.image-container img')?.complete,previous);
        assert.equal(await page.locator('#timeScale').inputValue(),timeValue==='linear'?'linear-a':timeValue);
        assert.equal(await page.locator('#spaceScale').inputValue(),spaceValue);
        assert.equal(await page.locator('#stepCount').textContent(),'1,001');
    }
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb2&spaceScale=exp&run=true`);
    await completed('6');
    assert.equal(await page.locator('#spaceScale').inputValue(),'linear');
    await page.goto(`${base}/time-scale-preview/index.html?case=${Math.random()}#program=bb2&timeScale=log&run=true`);
    await completed('6');
    assert.equal(await page.locator('#timeScale').inputValue(),'linear-a');
    assert.deepEqual(errors,[]);
    console.log('Browser checks passed: defaults, restart notice, live scale/palette, stop, halt, state colors, hash, Linear/1-Y X modes and all time modes.');
} finally {
    await browser.close();
}
