import { test, expect } from "@playwright/test";

const URL = "/index.html";

function watchErrors(page) {
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  return errors;
}

test("page loads with runs, comparisons, and committed data", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  await expect(page).toHaveTitle(/AIRE Trace Explorer/);
  await expect(page.locator("#tabs button")).toHaveCount(4);
  await expect(page.locator("#case-list details")).toHaveCount(26);
  const ok = await page.evaluate(() => ({
    versions: Object.keys(window.DATA.versions).length,
    comparisons: Object.keys(window.DATA.comparisons).length,
  }));
  expect(ok.versions).toBe(3);
  expect(ok.comparisons).toBe(3);
  expect(errors).toEqual([]);
});

test("version transition steps forward and backward", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  // the transition starts at the v1 baseline: no deltas apply
  await expect(page.locator("#val-ground")).toContainText("0.5682");
  await expect(page.locator("#vdeltas-wrap")).toBeHidden();
  // forward to v2: recall improved, no regressions, committed deltas shown
  await page.locator("#vt-step").click();
  await expect(page.locator("#val-ground")).toContainText("0.5682");
  await expect(page.locator("#vdeltas-wrap")).toBeVisible();
  await expect(page.locator("#vdeltas")).toContainText("improved 2");
  await expect(page.locator("#vdeltas")).toContainText("regressed 0");
  // forward to v3: groundedness collapses to 0 with +22 ungrounded
  await page.locator("#vt-step").click();
  await expect(page.locator("#bar-ground")).toHaveClass(/bad/);
  await expect(page.locator("#val-ground")).toHaveText("v3 fluent (planted regression): 0");
  await expect(page.locator("#vdeltas")).toContainText("+22");
  await expect(page.locator("#vdeltas")).toContainText("ungrounded");
  // backward: v2 again, then v1
  await page.locator("#vt-back").click();
  await expect(page.locator("#vdeltas")).toContainText("regressed 0");
  await page.locator("#vt-back").click();
  await expect(page.locator("#vdeltas-wrap")).toBeHidden();
  // reset restores the baseline
  await page.locator("#vt-step").click();
  await page.locator("#vt-reset").click();
  await expect(page.locator("#vdeltas-wrap")).toBeHidden();
  expect(errors).toEqual([]);
});

test("failure category filter shows exactly the committed cases", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  await page.locator("#fail-chips .chip[data-f='ungrounded']").click();
  await expect(page.locator("#case-list details")).toHaveCount(22);
  await page.locator("#fail-chips .chip[data-f='all']").click();
  await expect(page.locator("#case-list details")).toHaveCount(26);
  expect(errors).toEqual([]);
});

test("in-case walkthrough walks input to verdict", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  const firstCase = page.locator("#case-list details").first();
  await firstCase.locator("summary").click();
  await firstCase.locator(".wk-next").click();
  await firstCase.locator(".wk-next").click();
  await expect(firstCase.locator(".wk-pos")).toHaveText("stage 3 / 5");
  await firstCase.locator(".wk-next").click();
  await firstCase.locator(".wk-next").click();
  await expect(firstCase.locator(".wk-pos")).toHaveText("stage 5 / 5");
  await expect(firstCase.locator(".w-verdict")).toContainText("failure:");
  expect(errors).toEqual([]);
});

test("guided story tour opens and advances", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  await page.locator("#story-open").click();
  await expect(page.locator("#tour-card")).toBeVisible();
  await expect(page.locator("#tour-title")).toContainText("v3 sounds better");
  await page.locator("#tour-next").click();
  await expect(page.locator("#tour-pos")).toHaveText("step 2 / 5");
  await page.keyboard.press("Escape");
  await expect(page.locator("#tour-card")).toBeHidden();
  expect(errors).toEqual([]);
});

test("keyboard: arrows drive the active player", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(URL);
  // replay player is active by default
  await page.locator("#rp-next").click(); // focus inside #replay
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#rp-pos")).toHaveText("stage 3 / 7");
  await page.keyboard.press("ArrowLeft");
  await expect(page.locator("#rp-pos")).toHaveText("stage 2 / 7");
  // interacting with the version transition makes it active
  await page.locator("#vt-step").click();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#val-ground")).toContainText("0");
  await page.keyboard.press("ArrowLeft");
  await expect(page.locator("#val-ground")).toContainText("0.5682");
  expect(errors).toEqual([]);
});

test("no horizontal overflow at 390px", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(URL);
  const o = await page.evaluate(() => ({
    sw: document.documentElement.scrollWidth,
    w: document.documentElement.clientWidth,
  }));
  expect(o.sw).toBeLessThanOrEqual(o.w);
});
