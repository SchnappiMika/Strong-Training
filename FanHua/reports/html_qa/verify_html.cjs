const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/Schnappi SaMa/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function main() {
  const root = path.resolve(__dirname, '../../..');
  const reportDirectory = path.resolve(__dirname, '..');
  const checks = [];
  const errors = [];
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
    args: ['--no-sandbox'],
  });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await context.route('**/*', route => {
      const url = route.request().url();
      return url.startsWith('file:') || url.startsWith('data:') ? route.continue() : route.abort();
    });
    const tutorial = path.join(reportDirectory, '知识点教程_双游戏泛化.html');
    const meeting = path.join(reportDirectory, '组会汇报_双游戏泛化训练.html');
    for (const file of [tutorial, meeting]) {
      await page.goto(pathToFileURL(file).href);
      assert.equal(await page.locator('h1').count(), 1);
      for (const selector of ['body', '.hero', '.card', '.svgbox', 'details', '.callout', '.output', '.walkthrough', '.stage-panel']) {
        const styles = await page.locator(selector).evaluateAll(elements => elements.map(element => ({ background: getComputedStyle(element).backgroundColor, image: getComputedStyle(element).backgroundImage })));
        for (const style of styles) {
          assert.equal(style.background, 'rgb(255, 255, 255)', `${path.basename(file)} must have a white ${selector}`);
          assert.equal(style.image, 'none');
        }
      }
      if (file === meeting) {
        assert.doesNotMatch(await page.locator('body').innerText(), /可以这样解释给老师|讲稿|可以回答|建议汇报|老师可能追问|汇报时|照着讲|答辩式|建议组会/);
      }
      const links = await page.locator('a[href]').evaluateAll(anchors => anchors.map(anchor => anchor.getAttribute('href')));
      for (const href of links) {
        if (href.startsWith('#')) {
          assert.equal(await page.locator(href).count(), 1, `Missing anchor: ${href}`);
        } else if (!/^https?:/.test(href)) {
          await fs.access(path.resolve(reportDirectory, decodeURIComponent(href)));
        }
      }
      checks.push(`${path.basename(file)}: headings and local links`);
      checks.push(`${path.basename(file)}: white background and readable document panels`);
    }
    await page.goto(pathToFileURL(tutorial).href);
    await page.locator('[data-game="timberman"]').click();
    assert.match(await page.locator('#task-output').innerText(), /timberman/);
    await page.locator('[data-game="flappybird"]').click();
    await page.locator('#flap-up').click();
    assert.equal(await page.locator('#demo-bird').evaluate(element => element.style.top), '23%');
    await page.locator('#flap-down').click();
    assert.equal(await page.locator('#demo-bird').evaluate(element => element.style.top), '64%');
    for (const [identifier, value, output, expected] of [
      ['#gamma-slider', '100', '#gamma-output', '6.00'],
      ['#history-slider', '40', '#history-label', '40%'],
      ['#cal-q', '2', '#cal-output', '局部梯度为 0'],
      ['#cal-q', '12', '#cal-output', '局部梯度为 1'],
    ]) {
      await page.locator(identifier).evaluate((input, nextValue) => {
        input.value = nextValue;
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }, value);
      assert.match(await page.locator(output).innerText(), new RegExp(expected));
    }
    await page.locator('#gradient-step').click();
    assert.match(await page.locator('#gradient-output').innerText(), /w=0.400/);
    await page.locator('#gradient-reset').click();
    assert.match(await page.locator('#gradient-output').innerText(), /w=0.000/);
    for (const identifier of ['#latent-x', '#latent-y']) {
      await page.locator(identifier).evaluate(input => {
        input.value = '100';
        input.dispatchEvent(new Event('input', { bubbles: true }));
      });
    }
    assert.match(await page.locator('#latent-output').innerText(), /映射动作 1/);
    await page.locator('#sim-step').click();
    assert.match(await page.locator('#sim-explain').innerText(), /第 1 步/);
    await page.locator('#sim-reset').click();
    await page.locator('.quiz[data-answer="b"] [data-choice="b"]').click();
    assert.match(await page.locator('.quiz[data-answer="b"] .quiz-result').innerText(), /答对/);
    await page.locator('.quiz[data-answer="c"] [data-choice="c"]').click();
    assert.match(await page.locator('.quiz[data-answer="c"] .quiz-result').innerText(), /答对/);
    await page.locator('#motion-toggle').click();
    assert.equal(await page.locator('#motion-toggle').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#update-prev').isDisabled(), true);
    for (let step = 2; step <= 6; step++) {
      await page.locator('#update-next').click();
      assert.match(await page.locator('#update-output').innerText(), new RegExp(`第 ${step} / 6`));
    }
    assert.equal(await page.locator('#update-next').isDisabled(), true);
    await page.locator('#update-prev').click();
    assert.match(await page.locator('#update-output').innerText(), /8.58.*5.3824/);
    await page.locator('#update-reset').click();
    assert.match(await page.locator('#update-output').innerText(), /参数尚未改变/);
    checks.push('Worked update: all six steps, numeric example, boundaries, previous and reset');
    checks.push('Tutorial interactions: task, actions, discount, history, calibration, gradient, latent mapping, simulation, quizzes, pause');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(__dirname, 'tutorial_desktop.png') });
    await page.locator('#algorithm').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(__dirname, 'tutorial_algorithm.png') });
    await page.locator('#worked-update').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(__dirname, 'tutorial_update.png') });
    await page.goto(pathToFileURL(meeting).href);
    for (const stage of ['1', '2', '3', '4']) {
      await page.locator(`[data-stage="${stage}"]`).click();
      assert.match(await page.locator('#stage-panel').innerText(), new RegExp(`阶段 ${stage}`));
    }
    await page.locator('#presentation-toggle').click();
    assert.equal(await page.locator('body').evaluate(element => element.classList.contains('presentation')), true);
    await page.locator('#presentation-toggle').click();
    await page.locator('#notes-toggle').click();
    assert.equal(await page.locator('details:not([open])').count(), 0);
    await page.locator('#notes-toggle').click();
    assert.equal(await page.locator('details[open]').count(), 0);
    for (const task of ['flappybird', 'timberman']) {
      await page.locator(`[data-task="${task}"]`).click();
      assert.equal(await page.locator('#episode-chart circle').count(), 20);
      const run = task === 'flappybird' ? '20261008_172006_553338_evaluate_flappybird' : '20261008_200326_813205_evaluate_timberman';
      const original = JSON.parse(await fs.readFile(path.join(root, 'FanHua/experiments', run, 'evaluation.json'), 'utf8'));
      const titles = await page.locator('#episode-chart circle title').allTextContents();
      original.episodes.forEach((episode, index) => {
        const value = task === 'flappybird' ? episode.learning_return : episode.score;
        assert.match(titles[index], new RegExp(value.toFixed(4).replace('.', '\\.')));
        assert.equal(episode.steps, 3000);
        assert.equal(episode.terminated, false);
      });
      assert.equal(original.updates_before, original.updates_after);
      assert.equal(original.parameters_before, original.parameters_after);
    }
    checks.push('Meeting interactions: four stages, presentation, notes, 20-episode charts checked against original evaluations');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(__dirname, 'meeting_desktop.png') });
    await page.locator('#evidence').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(__dirname, 'meeting_results.png') });
    await page.locator('#method').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(__dirname, 'meeting_method.png') });
    for (const width of [390, 768, 1440]) {
      await page.setViewportSize({ width, height: 1000 });
      for (const file of [tutorial, meeting]) {
        await page.goto(pathToFileURL(file).href);
        const overflow = await page.evaluate(() => ({ width: window.innerWidth, scrollWidth: document.documentElement.scrollWidth }));
        assert.ok(overflow.scrollWidth <= overflow.width + 1, `${path.basename(file)} overflow at ${width}: ${JSON.stringify(overflow)}`);
        if (width === 390) {
          await page.screenshot({ path: path.join(__dirname, `${file === tutorial ? 'tutorial' : 'meeting'}_mobile.png`) });
        }
      }
      checks.push(`Responsive: no document overflow at ${width}px`);
    }
    await page.emulateMedia({ media: 'print' });
    for (const file of [tutorial, meeting]) {
      await page.goto(pathToFileURL(file).href);
      assert.equal(await page.locator('body').evaluate(element => getComputedStyle(element).color), 'rgb(17, 17, 17)');
      assert.equal(await page.locator('h3').first().evaluate(element => getComputedStyle(element).color), 'rgb(17, 17, 17)');
      const originalOpened = await page.locator('details[open]').count();
      await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
      assert.equal(await page.locator('details:not([open])').count(), 0);
      await page.evaluate(() => window.dispatchEvent(new Event('afterprint')));
      assert.equal(await page.locator('details[open]').count(), originalOpened);
    }
    checks.push('Print: light background, dark headings, full notes expanded and restored');
    assert.deepEqual(errors, []);
    const result = { verified_at: new Date().toISOString(), passed: true, checks, browser_errors: errors, training_or_game_started: false };
    await fs.writeFile(path.join(__dirname, 'verification.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
