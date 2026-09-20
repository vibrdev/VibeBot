"""Playwright wrapper: perception (what's on the page) and actuation (do the thing)."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    async_playwright,
)

from .config import BrowserConfig
from .schema import Element, Observation

log = logging.getLogger(__name__)

#: Runs in the page. Tags every interactive element with data-vb-idx and returns
#: a compact description of each, plus a text digest of the visible page.
_COLLECT_JS = r"""
(maxElements) => {
  const SELECTOR = [
    'a[href]', 'button', 'input', 'select', 'textarea', 'summary',
    '[role=button]', '[role=link]', '[role=checkbox]', '[role=radio]',
    '[role=tab]', '[role=menuitem]', '[role=option]', '[role=switch]',
    '[role=combobox]', '[role=searchbox]', '[role=textbox]',
    '[contenteditable=""]', '[contenteditable=true]', '[onclick]', '[tabindex]',
  ].join(',');

  const squash = (s, n) => (s || '').replace(/\s+/g, ' ').trim().slice(0, n);

  const labelFor = (el) => {
    if (el.id) {
      const lab = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (lab) return squash(lab.innerText, 120);
    }
    const wrapper = el.closest('label');
    if (wrapper) return squash(wrapper.innerText, 120);
    return '';
  };

  const visible = (el, rect) => {
    if (rect.width < 2 || rect.height < 2) return false;
    const style = getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') return false;
    if (parseFloat(style.opacity || '1') < 0.05) return false;
    if (el.disabled) return false;
    if (el.getAttribute('aria-hidden') === 'true') return false;
    return true;
  };

  document.querySelectorAll('[data-vb-idx]').forEach(el => el.removeAttribute('data-vb-idx'));

  const out = [];
  let idx = 0;
  for (const el of document.querySelectorAll(SELECTOR)) {
    if (out.length >= maxElements) break;
    const rect = el.getBoundingClientRect();
    if (!visible(el, rect)) continue;

    const tag = el.tagName.toLowerCase();
    const inViewport = rect.bottom > 0 && rect.top < innerHeight &&
                       rect.right > 0 && rect.left < innerWidth;

    el.setAttribute('data-vb-idx', String(idx));
    out.push({
      idx: idx++,
      tag,
      role: el.getAttribute('role') || '',
      text: squash(el.innerText || el.getAttribute('alt') || '', 120),
      name: squash(el.getAttribute('aria-label') || el.getAttribute('title') || labelFor(el), 120),
      placeholder: squash(el.getAttribute('placeholder') || '', 80),
      value: tag === 'input' && el.type === 'password' ? '' : squash(el.value || '', 60),
      href: squash(el.getAttribute('href') || '', 120),
      input_type: (el.getAttribute('type') || '').toLowerCase(),
      rect: [rect.x + scrollX, rect.y + scrollY, rect.width, rect.height],
      in_viewport: inViewport,
    });
  }

  const body = document.body ? squash(document.body.innerText, 4000) : '';
  return {
    url: location.href,
    title: document.title,
    elements: out,
    text: body,
    scroll: { y: Math.round(scrollY), height: Math.round(document.body ? document.body.scrollHeight : 0) },
  };
}
"""

#: Draws numbered boxes so a vision model can map what it sees to an index.
_ANNOTATE_JS = r"""
(indices) => {
  const wanted = new Set(indices.map(String));
  const layer = document.createElement('div');
  layer.id = '__vb_layer';
  Object.assign(layer.style, {
    position: 'fixed', inset: '0', zIndex: '2147483647', pointerEvents: 'none',
  });
  const palette = ['#ff3b30', '#0a84ff', '#34c759', '#ff9f0a', '#bf5af2', '#00c7be'];
  document.querySelectorAll('[data-vb-idx]').forEach((el) => {
    const idx = el.getAttribute('data-vb-idx');
    if (wanted.size && !wanted.has(idx)) return;
    const r = el.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) return;
    const color = palette[Number(idx) % palette.length];
    const box = document.createElement('div');
    Object.assign(box.style, {
      position: 'fixed', left: `${r.x}px`, top: `${r.y}px`,
      width: `${r.width}px`, height: `${r.height}px`,
      border: `2px solid ${color}`, borderRadius: '2px', boxSizing: 'border-box',
    });
    const tag = document.createElement('div');
    tag.textContent = idx;
    Object.assign(tag.style, {
      position: 'fixed', left: `${Math.max(0, r.x)}px`, top: `${Math.max(0, r.y - 16)}px`,
      background: color, color: '#fff', font: '700 11px/14px ui-monospace,monospace',
      padding: '0 4px', borderRadius: '2px',
    });
    layer.appendChild(box);
    layer.appendChild(tag);
  });
  document.documentElement.appendChild(layer);
}
"""

_DEANNOTATE_JS = "() => document.getElementById('__vb_layer')?.remove()"


class BrowserSession:
    """Owns the Playwright context. One goal run == one session."""

    def __init__(self, cfg: BrowserConfig):
        self.cfg = cfg
        self._pw = None
        self._browser: Browser | None = None
        self._ctx: BrowserContext | None = None
        self.page: Page | None = None

    async def start(self) -> None:
        headless = self.cfg.headless or _no_display()
        if headless and not self.cfg.headless:
            log.info("No display detected — running headless.")

        self._pw = await async_playwright().start()
        profile = Path(self.cfg.user_data_dir).expanduser()
        profile.mkdir(parents=True, exist_ok=True)

        launch_kwargs = dict(
            headless=headless,
            viewport={"width": self.cfg.width, "height": self.cfg.height},
            locale=self.cfg.locale,
            args=["--disable-blink-features=AutomationControlled"],
        )
        executable = os.environ.get("VIBEBOT_CHROMIUM_PATH")
        if executable:
            launch_kwargs["executable_path"] = executable

        # Persistent context keeps you logged into sites between runs.
        self._ctx = await self._pw.chromium.launch_persistent_context(str(profile), **launch_kwargs)
        self._ctx.set_default_timeout(self.cfg.action_timeout_ms)
        self._ctx.set_default_navigation_timeout(self.cfg.nav_timeout_ms)
        self.page = self._ctx.pages[0] if self._ctx.pages else await self._ctx.new_page()
        if self.cfg.start_url and self.cfg.start_url != "about:blank":
            await self.goto(self.cfg.start_url)

    async def close(self) -> None:
        for closer in (self._ctx, self._browser):
            try:
                if closer:
                    await closer.close()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass
        if self._pw:
            await self._pw.stop()

    # ------------------------------------------------------------- perception

    async def observe(self, max_elements: int = 150, screenshot: bool = True) -> Observation:
        page = self._live_page()
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=self.cfg.nav_timeout_ms)
        except PlaywrightError:
            pass
        await asyncio.sleep(0.25)  # let late-rendering frameworks settle

        raw = await page.evaluate(_COLLECT_JS, max_elements)
        elements = [
            Element(
                idx=item["idx"],
                tag=item["tag"],
                role=item["role"],
                text=item["text"],
                name=item["name"],
                placeholder=item["placeholder"],
                value=item["value"],
                href=item["href"],
                input_type=item["input_type"],
                rect=tuple(item["rect"]),
                in_viewport=item["in_viewport"],
            )
            for item in raw["elements"]
        ]
        shot = await self.screenshot() if screenshot else None
        return Observation(
            url=raw["url"],
            title=raw["title"],
            elements=elements,
            text_digest=raw["text"],
            screenshot_png=shot,
        )

    async def screenshot(self, highlight: list[int] | None = None) -> bytes | None:
        page = self._live_page()
        try:
            if highlight is not None:
                await page.evaluate(_ANNOTATE_JS, highlight)
            return await page.screenshot(type="png", full_page=False)
        except PlaywrightError as exc:
            log.warning("screenshot failed: %s", exc)
            return None
        finally:
            if highlight is not None:
                try:
                    await page.evaluate(_DEANNOTATE_JS)
                except PlaywrightError:
                    pass

    # ------------------------------------------------------------- actuation

    async def click(self, idx: int) -> str:
        locator = self._by_idx(idx)
        await locator.scroll_into_view_if_needed(timeout=5000)
        try:
            await locator.click(timeout=self.cfg.action_timeout_ms)
        except PlaywrightError:
            # Overlays, animations, cookie walls — fall back to a synthetic click.
            await locator.dispatch_event("click")
        await self._settle()
        return f"clicked element {idx}"

    async def type_text(self, idx: int, text: str, submit: bool = True) -> str:
        locator = self._by_idx(idx)
        await locator.scroll_into_view_if_needed(timeout=5000)
        await locator.click(timeout=self.cfg.action_timeout_ms)
        try:
            await locator.fill("")
        except PlaywrightError:
            pass
        await locator.type(text, delay=18)
        if submit:
            await locator.press("Enter")
        await self._settle()
        return f"typed {text!r} into element {idx}"

    async def select(self, idx: int, value: str) -> str:
        locator = self._by_idx(idx)
        try:
            await locator.select_option(label=value, timeout=self.cfg.action_timeout_ms)
        except PlaywrightError:
            await locator.select_option(value=value, timeout=self.cfg.action_timeout_ms)
        await self._settle()
        return f"selected {value!r} in element {idx}"

    async def scroll(self, amount: str = "down") -> str:
        page = self._live_page()
        delta = {"down": 0.85, "up": -0.85, "top": None, "bottom": None}.get(amount, 0.85)
        if amount == "top":
            await page.evaluate("() => scrollTo({top: 0})")
        elif amount == "bottom":
            await page.evaluate("() => scrollTo({top: document.body.scrollHeight})")
        else:
            await page.mouse.wheel(0, int(self.cfg.height * float(delta)))
        await asyncio.sleep(0.4)
        return f"scrolled {amount}"

    async def goto(self, url: str) -> str:
        page = self._live_page()
        if not urlparse(url).scheme:
            url = "https://" + url
        await page.goto(url, wait_until="domcontentloaded")
        await self._settle()
        return f"navigated to {url}"

    async def back(self) -> str:
        await self._live_page().go_back(wait_until="domcontentloaded")
        await self._settle()
        return "went back"

    async def wait(self, seconds: float = 1.5) -> str:
        await asyncio.sleep(min(seconds, 10))
        return f"waited {seconds}s"

    async def element_is_password(self, idx: int) -> bool:
        try:
            handle = self._by_idx(idx)
            return (await handle.get_attribute("type") or "").lower() == "password"
        except PlaywrightError:
            return False

    # ---------------------------------------------------------------- helpers

    def _live_page(self) -> Page:
        if self._ctx and self._ctx.pages:
            # A click may have opened a new tab; always act on the newest one.
            self.page = self._ctx.pages[-1]
        if self.page is None:
            raise RuntimeError("browser session not started")
        return self.page

    def _by_idx(self, idx: int):
        return self._live_page().locator(f'[data-vb-idx="{idx}"]').first

    async def _settle(self) -> None:
        page = self._live_page()
        try:
            await page.wait_for_load_state("networkidle", timeout=4000)
        except PlaywrightError:
            await asyncio.sleep(0.4)


def _no_display() -> bool:
    if os.name == "nt" or sys_platform_is_mac():
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def sys_platform_is_mac() -> bool:
    import sys

    return sys.platform == "darwin"
