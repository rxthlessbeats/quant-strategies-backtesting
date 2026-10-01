// Run against the running frontend/backend: NODE_PATH=/path/to/browser-tools/node_modules node scripts/check-redesign.cjs
// Browser tools: playwright and @axe-core/playwright. No application dependency is added.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const ts = require('../frontend/node_modules/typescript');
const base = process.env.TRADING_URL || 'http://127.0.0.1:3000';
const output = process.env.TRADING_REVIEW_DIR || '/tmp/trading-review';
fs.mkdirSync(output, { recursive: true });

function loadUtility(file) {
  const compiled = ts.transpileModule(fs.readFileSync(path.join(__dirname, '../frontend/src/lib', file), 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
  const exports = {};
  new Function('exports', compiled)(exports);
  return exports;
}
const { chartViewStart } = loadUtility('chart-view.ts');
const { formatDailyMarketAsOf } = loadUtility('market-timestamps.ts');
const { parseIndicatorSelections, buildIndicatorsQuery, selectionApiKey, buildColorMap, indicatorPane } = loadUtility('indicator-utils.ts');
assert.deepEqual(parseIndicatorSelections(''), []);
assert.equal(buildIndicatorsQuery(parseIndicatorSelections('sma:5,rsi:14')), 'sma:5,rsi:14');
assert.equal(buildIndicatorsQuery(parseIndicatorSelections('obv,ad')), 'obv,ad');
assert.equal(buildIndicatorsQuery(parseIndicatorSelections('macd')), 'macd:fast=12;slow=26;signal=9');
const stochastic = parseIndicatorSelections('stoch:smooth_d=2;period=14;smooth_k=3')[0];
assert.equal(selectionApiKey(stochastic), 'stoch_14_3_2');
assert.equal(buildIndicatorsQuery([stochastic]), 'stoch:period=14;smooth_k=3;smooth_d=2');
assert.notEqual(buildColorMap([stochastic]).stoch_k_14_3_2, buildColorMap([stochastic]).stoch_d_14_3_2);
assert.equal(indicatorPane('stoch_d_14_3_2'), 'stoch');
assert.equal(indicatorPane('donchian_upper_20'), null);
const timestamp = value => Date.parse(value) / 1000;
assert.equal(chartViewStart(timestamp('2026-05-31T13:30:00Z'), '3M'), timestamp('2026-02-28T13:30:00Z'));
assert.equal(chartViewStart(timestamp('2024-05-31T13:30:00Z'), '3M'), timestamp('2024-02-29T13:30:00Z'));
assert.equal(chartViewStart(timestamp('2026-09-30T13:30:00Z'), '1Y'), timestamp('2025-09-30T13:30:00Z'));
assert.equal(chartViewStart(timestamp('2026-09-30T13:30:00Z'), 'ALL'), 0);
assert.equal(formatDailyMarketAsOf(timestamp('2026-09-30T00:00:00Z')), 'Sep 30, 2026');
assert.equal(formatDailyMarketAsOf(NaN), null);

(async () => {
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  const responses = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error' && !message.text().includes('Failed to load resource')) console.error('BROWSER:', message.text()); });
  page.on('response', response => { if (response.url().includes('/api/backend/')) responses.push({ url: response.url().replace(base, ''), status: response.status() }); });
  page.setDefaultTimeout(60000);
  async function goto(route) {
    const response = await page.goto(base + route, { waitUntil: 'networkidle' });
    assert.equal(response.status(), 200, route);
  }
  async function ready(route) {
    if (route === '/') await page.locator('.landscape-chart').waitFor();
    if (route.startsWith('/chart')) { await page.locator('.quote-price').waitFor(); await page.locator('.chart-canvas canvas').first().waitFor(); }
  }
  async function research() {
    await page.locator('#analysts').scrollIntoViewIfNeeded();
    await page.locator('#analysts').getByRole('heading', { name: 'Analyst Recommendations', exact: true }).waitFor();
    await page.locator('#analysts canvas').first().waitFor();
    await page.locator('#financials').scrollIntoViewIfNeeded();
    await page.locator('#financials canvas').first().waitFor();
    await page.waitForTimeout(500);
  }
  async function capture(name) { await page.evaluate(() => { document.activeElement?.blur(); scrollTo({ top: 0, behavior: 'instant' }); }); await page.waitForTimeout(100); await page.screenshot({ path: path.join(output, name + '.png'), fullPage: true }); }
  async function audit(name) {
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, name + ' overflow');
    const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa']).analyze();
    const violations = audit.violations.map(item => ({ id: item.id, targets: item.nodes.map(node => node.target) }));
    fs.writeFileSync(path.join(output, name + '-audit.json'), JSON.stringify(violations, null, 2));
    assert.deepEqual(violations, [], name + ' accessibility');
  }
  async function candleGeometry() {
    return page.locator('.chart-canvas canvas').evaluateAll(canvases => canvases.map(canvas => {
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let count = 0, hash = 0;
      for (let index = 0; index < pixels.length; index += 4) {
        if ((pixels[index] === 38 && pixels[index + 1] === 166 && pixels[index + 2] === 154) || (pixels[index] === 239 && pixels[index + 1] === 83 && pixels[index + 2] === 80)) { count++; hash = (hash + index) >>> 0; }
      }
      return { width: canvas.width, height: canvas.height, count, hash };
    }));
  }
  async function chartResponse(action, includes) {
    const pending = page.waitForResponse(response => response.url().includes('/analysis/chart?') && (!includes || decodeURIComponent(response.url()).includes(includes)));
    await action();
    const response = await pending;
    assert.equal(response.status(), 200, 'chart API');
    const data = await response.json();
    await page.locator('.chart-toolbar .animate-spin').waitFor({ state: 'hidden' });
    return data;
  }
  try {
    const initialWorkspace = await page.request.get(base + '/chart?symbol=AAPL');
    assert.equal(initialWorkspace.status(), 200);
    assert.match(await initialWorkspace.text(), /<h1>AAPL<\/h1>/, 'ticker heading is server rendered');
    const marketResponse = page.waitForResponse(response => response.url().includes('/analysis/chart?symbol=SPY'));
    await goto('/'); await ready('/');
    const market = await (await marketResponse).json();
    const bars = market.bars.filter(bar => Number.isFinite(bar.close));
    const latest = bars.at(-1);
    assert.equal(await page.locator('.lens-price').textContent(), '$' + latest.close.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
    await page.getByRole('button', { name: '1M', exact: true }).click();
    const month = bars.filter(bar => bar.timestamp >= chartViewStart(latest.timestamp, '1M'));
    assert.equal((await page.locator('.price-trace').getAttribute('d')).match(/L/g).length + 1, month.length, 'calendar range');
    const timeline = page.getByRole('slider', { name: 'Price history timeline' });
    await timeline.focus(); await timeline.press('Home');
    assert.equal(await timeline.inputValue(), '1');
    assert.match(await timeline.getAttribute('aria-valuetext'), new RegExp(month[0].close.toFixed(2).replace('.', '\\.')));
    await timeline.press('End'); await page.getByRole('button', { name: 'Replay price history' }).click();
    await page.waitForTimeout(300);
    assert.ok(Number(await timeline.inputValue()) > 1 && Number(await timeline.inputValue()) < 100, 'replay advances');
    await page.getByRole('button', { name: 'Pause price replay' }).click();
    const stopped = await timeline.inputValue(); await page.waitForTimeout(200); assert.equal(await timeline.inputValue(), stopped, 'pause stops');
    await timeline.press('End');
    const nvdaResponse = page.waitForResponse(response => response.url().includes('/analysis/chart?symbol=NVDA'));
    await page.getByRole('button', { name: 'NVDA', exact: true }).click(); await nvdaResponse; await page.locator('.landscape-chart').waitFor();
    assert.equal(await page.locator('.lens-company').textContent(), 'NVIDIA');
    await audit('overview-light'); await capture('overview-desktop');

    await page.getByRole('combobox', { name: 'Search US ticker or company' }).fill('NVDA');
    await page.getByRole('option').first().waitFor();
    await page.getByRole('combobox').press('ArrowDown'); await page.getByRole('combobox').press('Enter');
    await page.waitForURL('**/chart?**'); await ready('/chart');
    assert.equal(await page.locator('.company-title h1').textContent(), 'NVDA');
    await page.getByRole('button', { name: '1M', exact: true }).click();
    assert.equal(await page.getByRole('button', { name: '1M', exact: true }).getAttribute('aria-pressed'), 'true');
    await page.getByRole('button', { name: 'Add indicator' }).click(); await page.getByRole('menuitem', { name: 'rsi', exact: true }).click();
    await page.getByLabel('Period', { exact: true }).fill('0'); await page.getByRole('button', { name: 'Apply', exact: true }).click();
    await page.getByRole('alert').filter({ hasText: 'positive numbers' }).waitFor();
    await page.getByLabel('Period', { exact: true }).fill('14');
    let updated = await chartResponse(() => page.getByRole('button', { name: 'Apply', exact: true }).click(), 'rsi');
    assert.ok(Object.keys(updated.indicators).some(key => key.startsWith('rsi_')), 'RSI series preserved');
    await page.getByRole('button', { name: 'Add indicator' }).click(); await page.getByRole('menuitem', { name: 'rsi', exact: true }).click();
    await page.getByRole('button', { name: 'Apply', exact: true }).click(); await page.getByText('Already on chart', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Back', exact: true }).click();
    await page.getByRole('button', { name: 'Settings for rsi', exact: true }).click(); await page.getByLabel('Period', { exact: true }).fill('21');
    await chartResponse(() => page.getByRole('button', { name: 'Apply', exact: true }).click(), 'rsi:21');
    await page.getByRole('button', { name: 'Save', exact: true }).click(); await page.getByText('Preset saved', { exact: true }).waitFor();
    assert.match(await page.evaluate(() => localStorage.getItem('rookie-trader-chart-settings')), /rsi:21/);
    await goto('/chart'); await ready('/chart'); await page.getByRole('button', { name: 'Settings for rsi', exact: true }).waitFor();
    await chartResponse(() => page.getByRole('button', { name: 'Remove rsi', exact: true }).click());
    assert.equal(await page.getByRole('button', { name: 'Settings for rsi', exact: true }).count(), 0);
    assert.equal(await page.getByRole('button', { name: 'Compare 1Y returns', exact: true }).getAttribute('aria-pressed'), 'true', 'default comparison period');
    const benchmark = page.getByLabel('Custom benchmark symbol');
    const comparisonResponse = page.waitForResponse(response => response.url().includes('/performance-comparison/NVDA?benchmark=QQQ'));
    await benchmark.fill('QQQ'); await page.getByRole('button', { name: 'Set custom benchmark' }).click();
    const comparison = await comparisonResponse;
    assert.equal(comparison.status(), 200, 'benchmark request');
    const comparisonData = await comparison.json();
    await page.locator('#performance').getByText('QQQ', { exact: true }).first().waitFor();
    const monthReturn = comparisonData.periods.find(period => period.label === '1M');
    await page.getByRole('button', { name: 'Compare 1M returns', exact: true }).click();
    assert.equal(await page.getByRole('button', { name: 'Compare 1M returns', exact: true }).getAttribute('aria-pressed'), 'true');
    const percent = value => value === null ? 'N/A' : (value > 0 ? '+' : '') + (value * 100).toFixed(2) + '%';
    assert.deepEqual(await page.locator('.comparison-values dd').allTextContents(), [percent(monthReturn.symbol_return), percent(monthReturn.benchmark_return)], 'selected comparison uses real returns');
    assert.match(await page.locator('.comparison-status').textContent(), /Showing 1M returns against QQQ/);
    for (const period of comparisonData.periods) {
      await page.getByRole('button', { name: `Compare ${period.label} returns`, exact: true }).click();
      const scale = Math.max(Math.abs(period.symbol_return ?? 0), Math.abs(period.benchmark_return ?? 0), .01);
      const bars = await page.locator('.return-track > span').evaluateAll(elements => elements.map(element => ({ width: parseFloat(element.style.width), left: parseFloat(element.style.left) })));
      [period.symbol_return, period.benchmark_return].forEach((value, index) => {
        const width = value === null ? 0 : Math.abs(value) / scale * 47;
        assert.ok(Math.abs(bars[index].width - width) < .001, 'bar width uses actual return');
        assert.ok(Math.abs(bars[index].left - (value !== null && value < 0 ? 50 - width : 50)) < .001, 'negative returns appear left of zero');
      });
    }
    await page.getByRole('button', { name: 'Compare 1M returns', exact: true }).click();

    await research();
    const meanTarget = await page.locator('.target-focus-value').textContent();
    await page.getByRole('button', { name: 'Current price', exact: true }).click();
    assert.notEqual(await page.locator('.target-focus-value').textContent(), meanTarget, 'target focus changes');
    assert.equal(await page.locator('.current-marker').getAttribute('data-active'), 'true');
    await page.getByRole('button', { name: 'Mean target', exact: true }).click();
    assert.equal(await page.locator('.financial-statement').count(), 5, 'all financial statement groups retained');
    const statement = page.locator('.financial-statement').filter({ has: page.getByRole('heading', { name: 'Income Statement', exact: true }) });
    const collapsedHeight = await statement.evaluate(element => element.offsetHeight);
    await statement.locator('summary').focus(); await statement.locator('summary').press('Enter');
    assert.equal(await statement.evaluate(element => element.open), true, 'financial statement opens from keyboard');
    await page.waitForTimeout(140);
    const halfwayHeight = await statement.evaluate(element => element.offsetHeight);
    await page.waitForTimeout(300);
    const expandedHeight = await statement.evaluate(element => element.offsetHeight);
    assert.ok(expandedHeight > collapsedHeight, 'statement reveals its metrics');
    if (await page.evaluate(() => CSS.supports('interpolate-size', 'allow-keywords') && CSS.supports('selector(::details-content)'))) assert.ok(halfwayHeight < expandedHeight, 'statement expansion animates');
    await statement.getByText('Revenue (ttm)', { exact: true }).waitFor();
    await statement.locator('summary').click();
    await page.getByRole('button', { name: 'Revenue vs earnings', exact: true }).click();
    await page.locator('.earnings-chart-stage canvas').first().waitFor();
    assert.match(await page.locator('.chart-feedback').textContent(), /revenue vs earnings/);
    await page.getByRole('button', { name: 'Cash vs debt', exact: true }).click();
    await page.locator('.earnings-chart-stage canvas').first().waitFor();
    await page.getByRole('button', { name: 'EPS estimate vs actual', exact: true }).click();
    await page.getByRole('button', { name: 'yearly', exact: true }).click(); await page.waitForTimeout(300);
    assert.equal(await page.getByRole('button', { name: 'yearly', exact: true }).getAttribute('aria-pressed'), 'true');
    await page.getByRole('button', { name: 'quarterly', exact: true }).click();
    await audit('workspace-light'); await capture('workspace-desktop');

    // An empty indicator preset must remain empty through navigation and reload.
    while (await page.getByRole('button', { name: 'Remove sma', exact: true }).count()) await chartResponse(() => page.getByRole('button', { name: 'Remove sma', exact: true }).first().click());
    await page.getByRole('button', { name: 'Save', exact: true }).click();
    await goto('/chart'); await ready('/chart');
    assert.equal(await page.getByRole('button', { name: /^Remove / }).count(), 0, 'empty preset restored');
    assert.equal(new URL(page.url()).searchParams.get('indicators'), '', 'empty preset deep link');
    await page.evaluate(() => localStorage.removeItem('rookie-trader-chart-settings'));

    await goto('/chart?symbol=NVDA&indicators='); await ready('/chart');
    await page.getByRole('button', { name: 'Add indicator' }).click();
    const indicatorMenu = page.getByRole('menu');
    assert.equal(await indicatorMenu.getByRole('menuitem').count(), 21);
    for (const category of ['trend', 'momentum', 'volatility', 'volume']) await indicatorMenu.getByRole('group', { name: category, exact: true }).waitFor();
    await indicatorMenu.getByRole('menuitem', { name: 'obv', exact: true }).click();
    await page.getByText('No parameters needed. Apply to add this indicator.', { exact: true }).waitFor();
    updated = await chartResponse(() => page.getByRole('button', { name: 'Apply', exact: true }).click(), 'obv');
    assert.deepEqual(Object.keys(updated.indicators), ['obv']);
    await page.getByRole('button', { name: 'Add indicator' }).click();
    await page.getByRole('menuitem', { name: 'stoch', exact: true }).click();
    await page.getByLabel('Smooth k', { exact: true }).fill('2.5');
    await page.getByRole('button', { name: 'Apply', exact: true }).click();
    await page.getByRole('alert').filter({ hasText: 'whole numbers' }).waitFor();
    await page.getByLabel('Smooth k', { exact: true }).fill('3');
    await page.getByLabel('Smooth d', { exact: true }).fill('2');
    updated = await chartResponse(() => page.getByRole('button', { name: 'Apply', exact: true }).click(), 'smooth_d=2');
    assert.deepEqual(Object.keys(updated.indicators).sort(), ['obv', 'stoch_d_14_3_2', 'stoch_k_14_3_2']);
    await page.getByRole('group', { name: /candlestick chart.*2 separate indicator panes/ }).waitFor();
    assert.equal(await page.locator('.chart-canvas > [role="group"]').evaluate(element => element.offsetHeight), 840);
    await page.getByRole('button', { name: 'Save', exact: true }).click();
    await goto('/chart'); await ready('/chart');
    await page.getByRole('button', { name: 'Settings for stoch', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Settings for obv', exact: true }).waitFor();
    await page.evaluate(() => localStorage.removeItem('rookie-trader-chart-settings'));

    await goto('/indicators'); const catalogCount = await page.locator('.indicator-item').count(); assert.equal(catalogCount, 21);
    const ids = await page.locator('.indicator-name h2').allTextContents();
    for (const [category, count] of [['trend', 6], ['momentum', 7], ['volatility', 4], ['volume', 4]]) {
      await page.getByRole('button', { name: category, exact: true }).click();
      assert.equal(await page.locator('.indicator-item').count(), count, category);
      assert.equal(await page.getByRole('status').textContent(), `${count} indicators in ${category}`);
    }
    await page.getByRole('button', { name: 'All signals', exact: true }).click();
    await page.getByRole('button', { name: /^momentum$/i }).click();
    assert.ok(await page.locator('.indicator-item').count() < catalogCount);
    await page.getByLabel('Search indicators').fill('RSI'); assert.equal(await page.locator('.indicator-item').count(), 1);
    await page.locator('.indicator-item summary').click(); await page.getByText('rsi:14', { exact: true }).waitFor();
    await audit('library-expanded'); await capture('library-expanded');
    await page.getByRole('link', { name: 'Explore in workspace' }).click(); await ready('/chart');
    assert.match(new URL(page.url()).searchParams.get('indicators'), /rsi:14/);
    const allIndicators = ids.map(id => id.toLowerCase()).join(',');
    await goto('/chart?' + new URLSearchParams({ symbol: 'NVDA', indicators: allIndicators })); await ready('/chart');
    await page.getByRole('group', { name: /28 indicator series in 13 separate indicator panes/ }).waitFor();
    assert.equal(await page.getByRole('button', { name: /^Settings for / }).count(), 21);
    assert.equal(await page.locator('.chart-canvas > [role="group"]').evaluate(element => element.offsetHeight), 2600);
    const paneHeights = await page.locator('.chart-canvas canvas').evaluateAll(canvases => canvases.filter(canvas => canvas.clientWidth > 500 && canvas.clientHeight > 40).map(canvas => canvas.clientHeight));
    assert.equal(paneHeights.filter(height => height > 450 && height < 550).length, 2, 'price pane keeps its height');
    assert.equal(paneHeights.filter(height => height > 140 && height < 180).length, 26, 'oscillator panes have room to read their signals');
    await audit('all-indicators'); await capture('all-indicators-desktop');
    await page.locator('.price-panel').screenshot({ path: path.join(output, 'indicator-panes-desktop.png') });
    await page.setViewportSize({ width: 390, height: 844 }); await audit('all-indicators-mobile'); await capture('all-indicators-mobile');
    await page.setViewportSize({ width: 1440, height: 1000 });
    await goto('/indicators'); await audit('indicators-light'); await capture('indicators-desktop');
    await goto('/health'); await page.getByRole('heading', { name: 'Ready when you are.' }).waitFor(); await audit('system-light'); await capture('system-desktop');

    for (const route of ['/', '/chart', '/indicators', '/health']) {
      await goto(route); await ready(route); if (route === '/chart') await research();
      let geometry;
      if (route === '/chart') {
        await page.locator('.chart-canvas').scrollIntoViewIfNeeded();
        await page.waitForTimeout(250);
        const beforeZoom = await candleGeometry();
        const box = await page.locator('.chart-canvas').boundingBox();
        await page.mouse.move(box.x + box.width / 2, box.y + 200); await page.mouse.wheel(0, -200); await page.waitForTimeout(250); await page.mouse.move(0, 0);
        geometry = await candleGeometry(); assert.ok(geometry.some(item => item.count > 0), 'candles drawn');
        assert.notDeepEqual(geometry, beforeZoom, 'wheel changes chart zoom');
      }
      await page.getByRole('button', { name: 'Switch to dark mode' }).click();
      if (geometry) { await page.waitForTimeout(250); assert.deepEqual(await candleGeometry(), geometry, 'theme keeps chart zoom and range'); }
      await page.waitForTimeout(350); await audit((route === '/' ? 'overview' : route.slice(1)) + '-dark'); await capture((route === '/' ? 'overview' : route.slice(1)) + '-dark');
      await page.getByRole('button', { name: 'Switch to light mode' }).click();
    }
    for (const width of [1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      for (const route of ['/', '/chart', '/indicators', '/health']) {
        await goto(route); await ready(route); if (route === '/chart') await research();
        await audit((route === '/' ? 'overview' : route.slice(1)) + '-' + width);
        if (width === 390 || width === 320) await capture((route === '/' ? 'overview' : route.slice(1)) + '-' + width);
        if (width === 390 && route === '/chart') for (const id of ['market-valuation', 'performance', 'analysts', 'financials']) await page.locator('#' + id).screenshot({ path: path.join(output, id + '-390.png') });
      }
    }
    // Failure and stale-search responses must not leave invented values or stale options.
    let unavailable = true;
    await page.route('**/api/backend/api/v1/analysis/chart?**', route => route.fulfill({ status: unavailable ? 503 : 200, contentType: 'application/json', body: JSON.stringify(unavailable ? { detail: 'Market data temporarily unavailable.' } : market) }));
    await goto('/'); await page.getByText('Price history is unavailable', { exact: true }).waitFor();
    assert.equal(await page.locator('.landscape-chart').count(), 0);
    assert.equal(await page.locator('.lens-price').textContent(), '—');
    await audit('overview-unavailable'); await capture('overview-unavailable');
    unavailable = false; await page.locator('.lens-empty').getByRole('button', { name: 'Try again' }).click(); await ready('/');
    await page.unroute('**/api/backend/api/v1/analysis/chart?**');
    await page.route('**/api/backend/api/v1/analysis/chart?**', route => route.abort('connectionfailed'));
    await goto('/chart?symbol=NVDA'); await page.getByRole('button', { name: 'Try again', exact: true }).waitFor();
    await page.unroute('**/api/backend/api/v1/analysis/chart?**');
    await page.getByRole('button', { name: 'Try again', exact: true }).click(); await ready('/chart');
    await goto('/'); await ready('/');
    await page.route('**/api/backend/api/v1/market/search?**', async route => {
      const value = new URL(route.request().url()).searchParams.get('keywords');
      if (value === 'OLD') await new Promise(resolve => setTimeout(resolve, 1200));
      await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ results: [{ symbol: value, name: value + ' company', type: 'EQUITY', region: 'NASDAQ' }] }) });
    });
    const oldRequest = page.waitForRequest(request => request.url().includes('keywords=OLD'));
    await page.getByRole('combobox').fill('OLD'); await oldRequest;
    await page.getByRole('combobox').fill('NEW'); await page.getByRole('option', { name: 'NEW NEW company' }).waitFor();
    await page.waitForTimeout(1400); assert.equal(await page.getByRole('option', { name: 'OLD OLD company' }).count(), 0, 'stale search discarded');
    await page.getByRole('combobox').press('Escape'); await page.unroute('**/api/backend/api/v1/market/search?**');
    assert.equal((await page.request.get(base + '/api/backend/health')).status(), 401, 'proxy rejects requests without same-origin browser context');
    await context.close();
    const reduced = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    const reducedPage = await reduced.newPage(); await reducedPage.goto(base); await reducedPage.locator('.landscape-chart').waitFor();
    // This context has no routing interception, which would disable browser caching.
    const cached = await reducedPage.evaluate(async () => {
      const url = '/api/backend/api/v1/analysis/chart?symbol=NVDA&interval=1d';
      const first = await fetch(url); const original = await first.json();
      const second = await fetch(url); const repeat = await second.json();
      const timing = performance.getEntriesByName(new URL(url, location.href).href).at(-1);
      const health = await fetch('/api/backend/health');
      const forced = await fetch('/api/backend/api/v1/market/data/NVDA/areas/statistics?force=false');
      const invalid = await fetch('/api/backend/api/v1/analysis/chart?symbol=NVDA&indicators=unknown');
      return { same: JSON.stringify(original) === JSON.stringify(repeat), count: repeat.bars.length,
        cache: second.headers.get('cache-control'), encoding: second.headers.get('content-encoding'),
        transferSize: timing.transferSize, health: health.headers.get('cache-control'),
        forced: forced.headers.get('cache-control'), invalid: invalid.headers.get('cache-control') };
    });
    assert.ok(cached.same && cached.count > 1000, 'cached price history preserves output');
    assert.equal(cached.cache, 'private, max-age=60');
    assert.equal(cached.encoding, 'gzip', 'price history is compressed');
    assert.equal(cached.transferSize, 0, 'repeat history loads from the browser cache');
    assert.equal(cached.health, 'no-store'); assert.equal(cached.forced, 'no-store'); assert.equal(cached.invalid, 'no-store');
    assert.ok(parseFloat(await reducedPage.locator('.price-trace').evaluate(element => getComputedStyle(element).animationDuration)) < .01, 'reduced motion respected');
    await reducedPage.screenshot({ path: path.join(output, 'reduced-motion.png') });
    await reducedPage.goto(base + '/chart'); await reducedPage.locator('.quote-price').waitFor();
    await reducedPage.locator('.return-track > span').first().waitFor();
    assert.ok(parseFloat(await reducedPage.locator('.return-track > span').first().evaluate(element => getComputedStyle(element).transitionDuration)) < .001, 'comparison motion respects reduced motion');
    await reduced.close();
    assert.deepEqual(errors, [], 'browser exceptions');
    for (const area of ['statistics', 'statements', 'earnings', 'analysts']) assert.ok(responses.some(response => response.url.includes('/api/v1/market/data/NVDA/areas/' + area) && response.status === 200), 'original company area: ' + area);
    assert.ok(responses.filter(response => response.url.includes('/api/v1/market/data/NVDA/areas/')).every(response => response.status === 200), 'original company areas');
    fs.writeFileSync(path.join(output, 'verification.json'), JSON.stringify({ result: 'passed', checks: ['calendar ranges', 'UTC dates', 'real price history', 'replay and pause', 'keyboard ticker search', 'indicator add/edit/remove', 'validation and duplicate protection', 'presets including empty', 'custom benchmarks', 'comparison periods and signed bars', 'target focus', 'keyboard statement expansion', 'earnings chart selectors', 'earnings views', 'catalog filtering and deep links', 'all original company areas', 'WCAG 2.2 AA', 'light/dark', '320/390/768/1024/1440px', 'reduced motion', 'failure recovery', 'stale search protection', 'no browser exceptions'], responses }, null, 2));
    console.log('PASS: redesign interactions, original API flows, responsive layouts, accessibility, and reduced motion.');
  } catch (error) {
    if (!page.isClosed()) { await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }); fs.writeFileSync(path.join(output, 'failure.txt'), await page.locator('body').innerText()); }
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
