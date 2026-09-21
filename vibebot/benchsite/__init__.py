"""A small, deterministic shop and wiki for the benchmark.

Real sites made the benchmark useless: eBay served bot checks to a headless
browser, listings changed between runs, and Google asked for a CAPTCHA. A
goal whose right answer moves cannot tell you whether a change helped.

This site never changes, and every goal on it has exactly one right answer.
It is built to contain the traps real runs actually fell into:

- the right listing is not on the first page under the default sort, so
  reading only the top of page one is not enough;
- a cheap listing says "1TB" - storage - next to 32GB of memory, the same
  confusion a real run made when it reported "1024GB RAM";
- a brand-new listing and a for-parts listing undercut the right answer, so
  "cheapest" without the condition is wrong;
- site furniture ("Deals", "Sign in", "Gift cards") sits on every page;
- the answer to the wiki question is thousands of characters down the page.
"""

from __future__ import annotations

from .server import CATALOG, SiteServer, start

__all__ = ["CATALOG", "SiteServer", "start"]
