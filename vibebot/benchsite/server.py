"""The benchmark site: a MacBook shop, a help page and a wiki article."""

from __future__ import annotations

import html
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse


@dataclass(frozen=True)
class Listing:
    id: str
    title: str
    chip: str
    memory: int
    storage: str
    condition: str
    price: int
    seller: str

    @property
    def family(self) -> str:
        return self.chip.split()[0]

    def spec(self) -> str:
        return f"{self.chip} · {self.memory}GB unified memory · {self.storage} SSD"


def sek(amount: int) -> str:
    return f"{amount:,} SEK"


# Listed in "best match" order. The right answer to the headline goal (used,
# at least 96GB, M-series, cheapest) is m2max-96-a at 21,450 SEK, which this
# order puts on page 3 of 3.
CATALOG: list[Listing] = [
    Listing("m4max-128-new", "MacBook Pro 16in M4 Max 128GB 2TB - Brand New Sealed", "M4 Max", 128, "2TB", "New", 20500, "applecenter_sthlm"),
    Listing("m1pro-32-1tb", "MacBook Pro 14in M1 Pro 32GB 1TB Space Grey", "M1 Pro", 32, "1TB", "Used", 9450, "techswap_gbg"),
    Listing("i9-64", "MacBook Pro 16in 2019 Intel Core i9 64GB 1TB", "Intel i9", 64, "1TB", "Used", 7900, "oldmac_dealer"),
    Listing("m3max-36-refurb", "MacBook Pro 14in M3 Max 36GB 1TB Refurbished", "M3 Max", 36, "1TB", "Refurbished", 16400, "greenbyte_refurb"),
    Listing("m2pro-16", "MacBook Pro 14in M2 Pro 16GB 512GB", "M2 Pro", 16, "512GB", "Used", 8900, "karlstad_it"),
    Listing("m4max-128-used-b", "MacBook Pro 16in M4 Max 128GB 4TB Space Black", "M4 Max", 128, "4TB", "Used", 34200, "proworkstations"),
    Listing("m1max-64", "MacBook Pro 16in M1 Max 64GB 2TB", "M1 Max", 64, "2TB", "Used", 14800, "malmo_macs"),
    Listing("m3pro-18", "MacBook Pro 14in M3 Pro 18GB 512GB Silver", "M3 Pro", 18, "512GB", "Used", 12300, "uppsala_electronics"),
    Listing("m2max-96-parts", "MacBook Pro 16in M2 Max 96GB 1TB - FOR PARTS, no display", "M2 Max", 96, "1TB", "For parts", 9900, "fixit_parts"),
    Listing("m4pro-48", "MacBook Pro 14in M4 Pro 48GB 1TB", "M4 Pro", 48, "1TB", "Used", 19800, "nordic_macs"),
    Listing("m3max-128-used", "MacBook Pro 16in M3 Max 128GB 2TB Space Black", "M3 Max", 128, "2TB", "Used", 28750, "proworkstations"),
    Listing("m2-24-air", "MacBook Air 15in M2 24GB 1TB", "M2", 24, "1TB", "Used", 8200, "linkoping_laptops"),
    Listing("m3max-96-refurb", "MacBook Pro 16in M3 Max 96GB 1TB Refurbished", "M3 Max", 96, "1TB", "Refurbished", 20990, "renewed_apple_se"),
    Listing("m1pro-16", "MacBook Pro 16in M1 Pro 16GB 512GB", "M1 Pro", 16, "512GB", "Used", 8600, "vasteras_tech"),
    Listing("m4max-36-new", "MacBook Pro 14in M4 Max 36GB 1TB - New", "M4 Max", 36, "1TB", "New", 33900, "applecenter_sthlm"),
    Listing("m2max-96-b", "MacBook Pro 16in M2 Max 96GB 1TB Silver", "M2 Max", 96, "1TB", "Used", 22300, "orebro_datorer"),
    Listing("i7-32", "MacBook Pro 15in 2018 Intel Core i7 32GB 512GB", "Intel i7", 32, "512GB", "Used", 5400, "oldmac_dealer"),
    Listing("m3max-96-used", "MacBook Pro 14in M3 Max 96GB 2TB", "M3 Max", 96, "2TB", "Used", 24900, "malmo_macs"),
    Listing("m2max-32", "MacBook Pro 16in M2 Max 32GB 1TB", "M2 Max", 32, "1TB", "Used", 15600, "techswap_gbg"),
    Listing("m4-16-air", "MacBook Air 13in M4 16GB 256GB", "M4", 16, "256GB", "New", 13990, "applecenter_sthlm"),
    Listing("m1max-32", "MacBook Pro 14in M1 Max 32GB 1TB", "M1 Max", 32, "1TB", "Used", 12900, "karlstad_it"),
    Listing("m4max-128-used", "MacBook Pro 14in M4 Max 128GB 2TB", "M4 Max", 128, "2TB", "Used", 31500, "nordic_macs"),
    Listing("m2max-96-a", "MacBook Pro 16in M2 Max 96GB 2TB Space Grey", "M2 Max", 96, "2TB", "Used", 21450, "nordic_macs"),
    Listing("m3pro-36-refurb", "MacBook Pro 16in M3 Pro 36GB 512GB Refurbished", "M3 Pro", 36, "512GB", "Refurbished", 17800, "greenbyte_refurb"),
    Listing("m2pro-32", "MacBook Pro 16in M2 Pro 32GB 1TB", "M2 Pro", 32, "1TB", "Used", 13400, "uppsala_electronics"),
    Listing("m1-16-air", "MacBook Air 13in M1 16GB 512GB", "M1", 16, "512GB", "Used", 5900, "vasteras_tech"),
    Listing("m3max-48", "MacBook Pro 16in M3 Max 48GB 1TB", "M3 Max", 48, "1TB", "Used", 23100, "orebro_datorer"),
    Listing("m4pro-24-refurb", "MacBook Pro 14in M4 Pro 24GB 512GB Refurbished", "M4 Pro", 24, "512GB", "Refurbished", 16900, "renewed_apple_se"),
    Listing("i9-32", "MacBook Pro 16in 2019 Intel Core i9 32GB 2TB", "Intel i9", 32, "2TB", "Used", 6800, "linkoping_laptops"),
    Listing("m1pro-32-b", "MacBook Pro 16in M1 Pro 32GB 1TB", "M1 Pro", 32, "1TB", "Used", 10400, "malmo_macs"),
]
BY_ID = {item.id: item for item in CATALOG}
PAGE_SIZE = 10

_STYLE = """
 body { font-family: system-ui, sans-serif; margin: 0; color: #222; }
 header { display: flex; gap: 14px; align-items: center; padding: 10px 18px; border-bottom: 1px solid #ddd; }
 header form { flex: 1; display: flex; gap: 6px; } header input { flex: 1; padding: 7px; }
 .promo { background: #fff4d6; padding: 8px 18px; font-size: 13px; }
 main { display: flex; gap: 24px; padding: 16px 18px; }
 aside { width: 200px; font-size: 14px; } aside a { display: block; margin: 3px 0; }
 .card { border: 1px solid #e3e3e3; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; }
 .price { font-size: 20px; font-weight: 700; } .cond { color: #666; }
 footer { border-top: 1px solid #ddd; padding: 14px 18px; font-size: 13px; }
"""


def _chrome(title: str, body: str, query: str = "") -> str:
    """Every page: a header full of things that are not the answer."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>"
        '<header><a href="/"><b>MacMarket</b></a>'
        '<form action="/search" method="get" role="search">'
        '<input name="q" type="search" placeholder="Search MacMarket" aria-label="Search MacMarket" '
        f'value="{html.escape(query)}"><button type="submit">Search</button></form>'
        '<a href="/deals">Deals</a> <a href="/gift-cards">Gift cards</a> '
        '<a href="/signin">Sign in</a> <a href="/register">Register</a> <a href="/cart">Cart (0)</a>'
        "</header>"
        '<div class="promo">Autumn sale: up to 30% off selected accessories. Free delivery on orders '
        "over 2,000 SEK. Trade in your old laptop for store credit. Members earn double points this "
        "week. Download our app for exclusive offers.</div>"
        f"{body}"
        '<footer><a href="/help/returns">Help &amp; returns</a> · <a href="/about">About MacMarket</a> · '
        '<a href="/wiki/halvardsen-tower">Local history wiki</a> · <a href="/careers">Careers</a> · '
        '<a href="/privacy">Privacy</a></footer></body></html>'
    )


def _search_url(params: dict[str, str], **changes: str) -> str:
    merged = {**params, **changes}
    if "page" not in changes:
        merged.pop("page", None)  # a new filter or sort starts again at page 1
    return "/search?" + urlencode({k: v for k, v in merged.items() if v})


def _filtered(params: dict[str, str]) -> list[Listing]:
    words = [w for w in params.get("q", "").lower().split() if w]
    items = [
        item for item in CATALOG
        if all(w in f"{item.title} {item.chip} {item.condition}".lower() for w in words)
    ]
    if params.get("cond"):
        items = [i for i in items if i.condition.lower() == params["cond"].lower()]
    if params.get("mem"):
        items = [i for i in items if i.memory == int(params["mem"])]
    if params.get("chip"):
        items = [i for i in items if i.family.lower() == params["chip"].lower()]
    sort = params.get("sort", "best")
    if sort == "price_asc":
        items = sorted(items, key=lambda i: i.price)
    elif sort == "price_desc":
        items = sorted(items, key=lambda i: -i.price)
    return items


def search_page(params: dict[str, str]) -> str:
    items = _filtered(params)
    try:
        page = max(1, int(params.get("page", "1") or 1))
    except ValueError:
        page = 1
    pages = max(1, -(-len(items) // PAGE_SIZE))
    shown = items[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    def facet(name: str, label: str, values: list[str]) -> str:
        # Active filters are toggles marked the way real shops mark them:
        # aria-current plus a visible tick. Clicking one turns it off.
        links = []
        for value in values:
            current = params.get(name, "") == value
            href = _search_url(params, **{name: "" if current else value})
            if current:
                links.append(f'<a href="{href}" aria-current="true"><b>{html.escape(value)} ✓</b></a>')
            else:
                links.append(f'<a href="{href}">{html.escape(value)}</a>')
        return f"<h4>{label}</h4>" + "".join(links)

    sorts = [("best", "Best match"), ("price_asc", "Price: lowest first"), ("price_desc", "Price: highest first")]
    sort_links = " · ".join(
        f'<a href="{_search_url(params, sort=key)}" aria-current="true"><b>{text}</b></a>'
        if params.get("sort", "best") == key
        else f'<a href="{_search_url(params, sort=key)}">{text}</a>'
        for key, text in sorts
    )
    cards = "".join(
        f'<div class="card"><a href="/item/{i.id}">{html.escape(i.title)}</a>'
        f"<div>{html.escape(i.spec())}</div><div class=\"cond\">{i.condition}</div>"
        f'<div class="price">{sek(i.price)}</div></div>'
        for i in shown
    ) or "<p>No listings match.</p>"
    nav = " ".join(
        f"<b>{n}</b>" if n == page else f'<a href="{_search_url(params, page=str(n))}">{n}</a>'
        for n in range(1, pages + 1)
    )
    if page < pages:
        nav += f' <a href="{_search_url(params, page=str(page + 1))}">Next page</a>'
    label = html.escape(params.get("q", "") or "all listings")
    body = (
        "<main><aside>"
        + facet("cond", "Condition", ["New", "Used", "Refurbished", "For parts"])
        + facet("mem", "Memory (RAM)", ["16", "24", "32", "36", "48", "64", "96", "128"])
        + facet("chip", "Chip", ["M1", "M2", "M3", "M4", "Intel"])
        + "</aside><section>"
        f"<p>{len(items)} results for <b>{label}</b> · page {page} of {pages}</p>"
        f"<p>Sort: {sort_links}</p>{cards}<p>Pages: {nav}</p></section></main>"
    )
    return _chrome(f"{params.get('q') or 'All listings'} - MacMarket search", body, params.get("q", ""))


def item_page(item: Listing) -> str:
    body = (
        f"<main><section><h1>{html.escape(item.title)}</h1>"
        f'<p class="price">{sek(item.price)}</p>'
        f"<p>Condition: <b>{item.condition}</b></p>"
        f"<table><tr><td>Chip</td><td>{item.chip}</td></tr>"
        f"<tr><td>Memory</td><td>{item.memory}GB unified memory</td></tr>"
        f"<tr><td>Storage</td><td>{item.storage} SSD</td></tr></table>"
        f"<p>Sold by <b>{item.seller}</b> · 98.9% positive feedback · ships from Sweden</p>"
        "<p><button>Buy It Now</button> <button>Add to cart</button> <button>Make an offer</button></p>"
        "</section></main>"
    )
    return _chrome(f"{item.title} - MacMarket", body)


def home_page() -> str:
    featured = [BY_ID["m4max-36-new"], BY_ID["m4-16-air"], BY_ID["m4max-128-new"]]
    cards = "".join(
        f'<div class="card"><a href="/item/{i.id}">{html.escape(i.title)}</a>'
        f'<div class="price">{sek(i.price)}</div></div>'
        for i in featured
    )
    body = (
        "<main><section><h1>Welcome to MacMarket</h1>"
        "<p>Sweden's marketplace for new, used and refurbished Macs. Browse "
        '<a href="/search?q=macbook">all MacBook listings</a> or search above.</p>'
        f"<h2>Featured today</h2>{cards}</section></main>"
    )
    return _chrome("MacMarket - new and used Macs", body)


def returns_page() -> str:
    body = (
        "<main><section><h1>Help &amp; returns</h1>"
        "<p>Shipping is handled by each seller. Most orders arrive within 2-4 working days.</p>"
        "<p>Payments are protected by MacMarket Buyer Guarantee. Contact the seller first if "
        "something is wrong.</p><h2>Returns</h2>"
        "<p>You can return any item within <b>30 days</b> of delivery for a full refund, as long as "
        'it is in the condition you received it in. Items sold "for parts" cannot be returned.</p>'
        "</section></main>"
    )
    return _chrome("Help & returns - MacMarket", body)


_TOWER_FILLER = (
    "The surrounding district grew quickly around the harbour, and the tower became a landmark for "
    "ships entering the bay. Local merchants financed several extensions to the quay, and the "
    "municipal council debated for years whether a permanent beacon should be built on the headland. "
)


def tower_page() -> str:
    early = "".join(f"<p>{_TOWER_FILLER}</p>" for _ in range(12))
    body = (
        "<main><section><h1>Halvardsen Tower</h1>"
        "<p>The Halvardsen Tower is a riveted iron lattice tower on the headland above Norrvik "
        "harbour. It is named after the engineer Ragnhild Halvardsen, whose design won the "
        f"municipal competition.</p><h2>Background</h2>{early}<h2>Construction</h2>"
        "<p>Foundations were laid in the spring of 1903. Work stopped twice for lack of funds, and "
        "the upper platform had to be redesigned after a storm damaged the scaffolding in 1905.</p>"
        "<p>The tower was finally completed in <b>1907</b>, and opened to visitors the following "
        "summer.</p><h2>Later history</h2><p>An electric beacon replaced the original oil lamp in "
        "1931. The tower was restored in 1988.</p></section></main>"
    )
    return _chrome("Halvardsen Tower - Local history wiki", body)


def filler_page(title: str) -> str:
    return _chrome(
        f"{title} - MacMarket",
        f"<main><section><h1>{html.escape(title)}</h1>"
        "<p>Nothing here will help you find a laptop.</p></section></main>",
    )


_FILLER_PATHS = {"/deals", "/gift-cards", "/signin", "/register", "/cart", "/about", "/careers", "/privacy"}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        path = parsed.path.rstrip("/") or "/"
        status = 200
        if path == "/":
            page = home_page()
        elif path == "/search":
            page = search_page(params)
        elif path.startswith("/item/") and path[6:] in BY_ID:
            page = item_page(BY_ID[path[6:]])
        elif path == "/help/returns":
            page = returns_page()
        elif path == "/wiki/halvardsen-tower":
            page = tower_page()
        elif path in _FILLER_PATHS:
            page = filler_page(path.strip("/").replace("-", " ").title())
        else:
            status, page = 404, filler_page("Page not found")
        data = page.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # noqa: ANN002 - keep bench output clean
        return


class SiteServer:
    def __init__(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="benchsite", daemon=True)

    def start(self) -> SiteServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def start() -> SiteServer:
    return SiteServer().start()
