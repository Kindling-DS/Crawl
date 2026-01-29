#!/usr/bin/env python3
"""
hibuddy_storecount_scraper_v6_6_full_big.py

Big/full version (keeps the richer crawling + export behavior).

Fixes / changes in this version
-------------------------------
1) SAME-CATEGORY BACK-TO-BACK BUG FIX
   - Previous SKU navigation goes into /product/... pages.
   - If the next SKU is in the SAME category and caching is enabled, the script now
     always returns to the category URL before scoring/matching (fast).
   - Category products are still cached (links + card text) so we don't need to re-scroll
     every time.

2) Robust two-input support (optional)
   - You can pass --input1 and/or --input2.
   - We only *require*:
       Sub Category (or Subcategory)
       ItemName (or SKU Description / Product Name)
     Brand/SKU are used when available but not required.

3) Multiple matches per SKU (captures different sizes)
4) Variant tabs support (7x0.5g, 14x0.5g, etc.) — extracts all variants
5) Saves store-level competitor pricing:
     StoreName + Price (+ RowText for debugging)

IMPORTANT
---------
You are responsible for complying with hibuddy.ca Terms/robots/policies.
Keep request rate low (delays are built in; increase if needed).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

HIBUDDY_HOME = "https://hibuddy.ca"
HIBUDDY_PRODUCTS_ROOT = "https://hibuddy.ca/products"

# -----------------------------
# Utils
# -----------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", norm(s))

def sleep(a: float = 0.6, b: float = 1.1):
    time.sleep(random.uniform(a, b))

def maybe_click_age_gate(page):
    try:
        btn = page.get_by_role("button", name=re.compile(r"^\s*yes\s*$", re.I))
        if btn.count() and btn.first.is_visible():
            btn.first.click()
    except Exception:
        pass

# -----------------------------
# Category routing
# -----------------------------
def category_url(subcat: Optional[str]) -> str:
    s = norm(subcat or "")
    if not s:
        return HIBUDDY_PRODUCTS_ROOT
    if "flower" in s:
        return f"{HIBUDDY_PRODUCTS_ROOT}/flowers"
    if "vape" in s or "cartridge" in s or "510" in s or "disposable" in s:
        return f"{HIBUDDY_PRODUCTS_ROOT}/vapes"
    if ("pre" in s and "roll" in s) or "pre-roll" in s or "preroll" in s:
        return f"{HIBUDDY_PRODUCTS_ROOT}/pre-rolls"
    if "edible" in s or "gummy" in s or "chocolate" in s or "beverage" in s:
        return f"{HIBUDDY_PRODUCTS_ROOT}/edibles"
    # Hibuddy: capsules/softgels live under concentrates/extracts
    if "capsule" in s or "capsules" in s or "softgel" in s or "softgels" in s:
        return f"{HIBUDDY_PRODUCTS_ROOT}/extracts"
    if "extract" in s or "concentrate" in s or "resin" in s or "rosin" in s or "shatter" in s or "oil" in s:
        return f"{HIBUDDY_PRODUCTS_ROOT}/extracts"
    if "topical" in s or "cream" in s or "balm" in s or "lotion" in s:
        return f"{HIBUDDY_PRODUCTS_ROOT}/topicals"
    return HIBUDDY_PRODUCTS_ROOT

# -----------------------------
# One-time location prompt
# -----------------------------
def ensure_location_once(page, interactive: bool, city: str, province: str, country: str, radius_km: int):
    if not interactive:
        return
    print("\n=== ONE-TIME SETUP ===")
    print("1) Accept age gate if prompted")
    print(f"2) Set location: {city}, {province}, {country}")
    print(f"3) Set radius: {radius_km} km")
    print("4) Ensure product grid is visible")
    input("Press ENTER when ready...\n")

# -----------------------------
# Input loading (robust; minimal required cols)
# -----------------------------
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lookup:
            return lookup[key]
    return None

def _load_one(path: str, source_tag: str) -> pd.DataFrame:
    # Limit columns for stability (many exports have far-right image/url columns)
    try:
        df = pd.read_excel(path, usecols="A:AZ")
    except Exception:
        df = pd.read_excel(path)

    subcat_c = _pick_col(df, ["Sub Category", "Subcategory"])
    name_c = _pick_col(df, ["SKU Description", "ItemName"])
    brand_c = _pick_col(df, ["Brand", "Brand Name"])
    sku_c = _pick_col(df, ["GTIN", "SKU"])

    if subcat_c is None or name_c is None:
        raise SystemExit(
            f"{source_tag} missing required columns. Need: "
            "Sub Category/Subcategory AND ItemName/SKU Description/Product Name"
        )

    out = pd.DataFrame({
        "SKU": df[sku_c] if sku_c else "",
        "Brand": df[brand_c] if brand_c else "",
        "ItemName": df[name_c],
        "Sub Category": df[subcat_c],
        "_source": source_tag,
    }).copy()

    # Clean
    for c in ["SKU", "Brand", "ItemName", "Sub Category"]:
        out[c] = out[c].astype(str).fillna("").str.strip()

    out = out[out["ItemName"].str.len() > 0]
    out = out[out["Sub Category"].str.len() > 0]
    return out

def load_inputs(input1: Optional[str], input2: Optional[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if input1:
        frames.append(_load_one(input1, "input1"))
    if input2:
        frames.append(_load_one(input2, "input2"))
    if not frames:
        raise SystemExit("Provide at least one input file: --input1 and/or --input2")
    return pd.concat(frames, ignore_index=True)

# -----------------------------
# Crawl ALL products in category (scroll)
# -----------------------------
def expand_all_products(page, max_scrolls: int = 90, stable_rounds: int = 3):
    last = 0
    stable = 0
    for _ in range(max_scrolls):
        n = page.locator('a[href^="/product/"]').count()
        if n == last:
            stable += 1
        else:
            stable = 0
            last = n
        if stable >= stable_rounds:
            break
        try:
            page.mouse.wheel(0, 2500)
        except Exception:
            page.evaluate("window.scrollBy(0, 2500)")
        sleep(0.45, 0.85)

def collect_product_links(page) -> List[str]:
    loc = page.locator('a[href^="/product/"]')
    hrefs: List[str] = []
    for i in range(loc.count()):
        try:
            h = loc.nth(i).get_attribute("href")
            if h and h.startswith("/product/"):
                hrefs.append(h)
        except Exception:
            pass
    seen = set()
    out: List[str] = []
    for h in hrefs:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out

def collect_product_cards(page) -> Dict[str, str]:
    loc = page.locator('a[href^="/product/"]')
    out: Dict[str, str] = {}
    for i in range(loc.count()):
        a = loc.nth(i)
        try:
            href = a.get_attribute("href") or ""
            if not href.startswith("/product/"):
                continue
            txt = (a.inner_text(timeout=800) or "").strip()
            if href and href not in out:
                out[href] = txt
        except Exception:
            continue
    return out

# -----------------------------
# Matching / scoring
# -----------------------------
@dataclass
class Expected:
    sku: str
    brand: str
    name: str

def strip_size(s: str) -> str:
    s = s or ""
    s = re.sub(r"\b\d+\s*[x×]\s*\d+(?:\.\d+)?\s*(g|ml)\b", " ", s, flags=re.I)
    s = re.sub(r"\b\d+(?:\.\d+)?\s*(g|ml)\b", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()

def score_text(text: str, exp: Expected) -> float:
    t = norm(text)
    b = norm(exp.brand)
    name = norm(strip_size(exp.name))

    sc = 0.0
    if b and b in t:
        sc += 25.0

    tt = set(tokens(text))
    bt = set(tokens(exp.brand))
    nt = set(tokens(strip_size(exp.name)))
    sc += 3.0 * len((bt | nt) & tt)

    try:
        import difflib
        sc += 18.0 * difflib.SequenceMatcher(None, name, t).ratio()
    except Exception:
        pass
    return sc

def extract_product_title(page) -> str:
    try:
        h1 = page.locator("h1").first
        if h1.count():
            return (h1.inner_text(timeout=1500) or "").strip()
    except Exception:
        pass
    return ""

def choose_matching_products(
    page,
    exp: Expected,
    links: List[str],
    verify_top_k: int,
    min_score: float,
    max_matches: int,
    allow_fallback: bool = False,
    card_texts: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, float, str]]:
    """
    Returns up to max_matches as (href, score, title).

    - Scores full category card texts (cached when available).
    - Filters above min_score + simple token sanity.
    - Verifies by opening product pages (limited).
    """
    scored: List[Tuple[float, str, str]] = []
    for h in links:
        txt = (card_texts or {}).get(h, "")
        if not txt:
            # fallback if needed (should rarely happen)
            try:
                a = page.locator(f'a[href="{h}"]').first
                if a.count():
                    txt = (a.inner_text(timeout=800) or "").strip()
            except Exception:
                txt = ""
        sc = score_text(txt, exp)
        scored.append((sc, h, txt))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return []

    exp_brand = norm(exp.brand)
    name_tokens = tokens(strip_size(exp.name))

    filtered: List[Tuple[float, str]] = []
    for sc, href, txt in scored:
        if sc < min_score:
            break
        t = norm(txt)
        if exp_brand and exp_brand not in t:
            continue
        if name_tokens:
            # require at least one of the first few tokens present
            txt_tokens = set(tokens(t))
            if not any(tok in txt_tokens for tok in name_tokens[:3]):
                continue
        filtered.append((sc, href))

    if not filtered:
        if not allow_fallback:
            return []
        # fallback (explicitly enabled): keep best few even if below threshold
        filtered = [(scored[i][0], scored[i][1]) for i in range(min(max_matches, len(scored)))]

    out: List[Tuple[str, float, str]] = []
    checked = 0
    for base_sc, href in filtered[:max_matches]:
        resc = base_sc
        title = ""
        try:
            page.goto(f"{HIBUDDY_HOME}{href}", wait_until="domcontentloaded")
            maybe_click_age_gate(page)
            sleep(0.65, 1.15)
            title = extract_product_title(page)
            if title:
                resc = score_text(title, exp) + 10.0
        except Exception:
            # keep base candidate if verify fails
            pass

        out.append((href, resc, title))
        checked += 1
        if checked >= verify_top_k * max_matches:
            break

    if not allow_fallback:
        out = [t for t in out if t[1] >= min_score]
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:max_matches]

# -----------------------------
# Variant tabs extraction
# -----------------------------
def get_variant_tabs(page) -> List[str]:
    out: List[str] = []
    try:
        tabs = page.locator("div.tabs a.tab")
        n = tabs.count()
        for i in range(min(n, 30)):
            t = (tabs.nth(i).inner_text(timeout=1200) or "").strip()
            if t and t not in out:
                out.append(t)
    except Exception:
        return out
    return out

def click_variant_tab(page, tab_text: str) -> bool:
    try:
        tabs = page.locator("div.tabs a.tab").filter(
            has_text=re.compile(rf"^\s*{re.escape(tab_text)}\s*$", re.I)
        )
        if tabs.count() == 0:
            return False
        el = tabs.first
        if el.is_visible():
            el.click()
            sleep(0.6, 1.1)
            return True
    except Exception:
        return False
    return False

# -----------------------------
# Retailer extraction
# -----------------------------
def extract_store_count(page) -> Optional[int]:
    try:
        body = page.inner_text("body", timeout=8000)
        m = re.search(r"\b(\d{1,4})\s+retailers?\s+within\b", body, flags=re.I)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None

def extract_price_value(text: str) -> Optional[float]:
    m = re.search(r"\$\s*([\d,]+(?:\.\d{2})?)", text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None

def select_view_all(page):
    try:
        sel = page.locator('select[aria-label="Rows per page:"]')
        if sel.count() == 0:
            return
        opts = sel.locator("option").all_text_contents()
        if any("VIEW ALL" in (o or "").upper() for o in opts):
            sel.select_option(label=re.compile(r"VIEW\s+ALL", re.I))
            sleep(0.6, 1.1)
            return
        nums = []
        for o in opts:
            m = re.search(r"\b(\d+)\b", o or "")
            if m:
                nums.append(int(m.group(1)))
        if nums:
            sel.select_option(label=str(max(nums)))
            sleep(0.6, 1.1)
    except Exception:
        pass

def extract_retailer_rows(page) -> List[Dict[str, Any]]:
    table = page.locator(".rdt_Table")
    if table.count() == 0:
        return []

    select_view_all(page)

    seen = set()
    out: List[Dict[str, Any]] = []

    def add_current():
        rows = page.locator(".rdt_TableBody .rdt_TableRow")
        for i in range(rows.count()):
            row = rows.nth(i)
            try:
                cells = row.locator(".rdt_TableCell")
                texts = [(cells.nth(j).inner_text(timeout=1200) or "").strip() for j in range(min(cells.count(), 8))]
                if not texts:
                    continue
                store = texts[0]
                price = None
                for t in texts[1:]:
                    p = extract_price_value(t)
                    if p is not None:
                        price = p
                        break
                key = (store, price)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"store_name": store, "price": price, "row_text": " | ".join(texts)})
            except Exception:
                continue

    add_current()

    for _ in range(150):
        nxt = page.locator('button[aria-label="Next Page"]')
        if nxt.count() == 0 or not nxt.first.is_enabled():
            break
        try:
            nxt.first.click()
            sleep(0.7, 1.2)
            add_current()
        except Exception:
            break

    return out

# -----------------------------
# Main
# -----------------------------
def run(args):
    df = load_inputs(args.input1, args.input2)

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip rows by (source, name) unless SKU present
    done = set()
    if out.exists() and args.resume:
        try:
            prev = pd.read_csv(out, dtype=str)
            if "SKU" in prev.columns and prev["SKU"].notna().any():
                done = set(prev["SKU"].astype(str).tolist())
        except Exception:
            pass

    header = [
        "SKU","Brand","ItemName","Sub Category","Source",
        "CategoryURL","HibuddyProductURL","Title",
        "MatchRank","MatchScore","RetailersWithinRadius",
        "VariantClass","StoreName","Price","RowText","Status"
    ]
    if not out.exists() or not args.resume:
        with out.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(Path(args.profile_dir).expanduser().resolve()),
            headless=args.headless,
            viewport={"width": 1500, "height": 950},
        )
        page = ctx.new_page()

        page.goto(HIBUDDY_HOME, wait_until="domcontentloaded")
        maybe_click_age_gate(page)
        page.goto(HIBUDDY_PRODUCTS_ROOT, wait_until="domcontentloaded")
        maybe_click_age_gate(page)

        ensure_location_once(page, args.interactive_setup, args.city, args.province, args.country, args.radius_km)

        active_cat: Optional[str] = None
        cached_links: List[str] = []
        cached_card_texts: Dict[str, str] = {}

        for _, r in df.iterrows():
            sku = r.get("SKU", "")
            brand = r.get("Brand", "")
            itemname = r.get("ItemName", "")
            subcat = r.get("Sub Category", "")
            source = r.get("_source", "")

            if args.resume and sku and sku in done:
                continue

            if not itemname.strip():
                # minimal fallback
                itemname = f"{brand} {sku}".strip() or sku or "UNKNOWN"

            url = category_url(subcat)
            print(f"\n🔎 CATEGORY [{subcat}] → {itemname}")

            # Load category (or rebuild cache)
            if (not args.cache_category_products) or (url != active_cat) or (not cached_links) or (not cached_card_texts):
                page.goto(url, wait_until="domcontentloaded")
                maybe_click_age_gate(page)
                sleep(0.9, 1.35)
                expand_all_products(page, max_scrolls=args.max_scrolls, stable_rounds=3)
                cached_links = collect_product_links(page)
                cached_card_texts = collect_product_cards(page)
                active_cat = url
                print(f"   Loaded products in category: {len(cached_links)}")
            else:
                # KEY FIX: even if category is same, ensure we are on the category page before matching.
                # This prevents the "2nd item in same category fails" issue.
                try:
                    if not page.url.startswith(url):
                        page.goto(url, wait_until="domcontentloaded")
                        maybe_click_age_gate(page)
                        sleep(0.6, 1.0)
                except Exception:
                    pass

            exp = Expected(sku=sku, brand=brand, name=itemname)

            status = "ok"
            match_outputs: List[Tuple[str, float, str, int, Optional[int], List[Dict[str, Any]]]] = []
            # tuple: (hibuddy_url, score, title, rank, store_count, rows)

            try:
                matches = choose_matching_products(
                    page,
                    exp,
                    cached_links,
                    verify_top_k=args.verify_top_k,
                    min_score=args.min_match_score,
                    max_matches=args.max_matches_per_sku,
                    allow_fallback=args.allow_fallback_below_threshold,
                    card_texts=cached_card_texts,
                )

                if not matches:
                    status = "no_match_in_category"
                else:
                    rank = 0
                    for href, sc, title in matches:
                        rank += 1
                        hibuddy_url = f"{HIBUDDY_HOME}{href}"

                        try:
                            page.goto(hibuddy_url, wait_until="domcontentloaded")
                            maybe_click_age_gate(page)
                            sleep(0.85, 1.45)
                        except Exception:
                            pass

                        store_count = extract_store_count(page)

                        # Variants
                        variant_tabs = get_variant_tabs(page)
                        if not variant_tabs:
                            rows = extract_retailer_rows(page)
                            for rr in rows:
                                rr["_variant_class"] = ""
                            match_outputs.append((hibuddy_url, sc, title, rank, store_count, rows))
                        else:
                            all_rows: List[Dict[str, Any]] = []
                            for vt in variant_tabs:
                                click_variant_tab(page, vt)
                                rows = extract_retailer_rows(page)
                                for rr in rows:
                                    rr["_variant_class"] = vt
                                all_rows.extend(rows)
                                sleep(0.35, 0.7)
                            match_outputs.append((hibuddy_url, sc, title, rank, store_count, all_rows))

                        sleep(0.4, 0.8)

            except PlaywrightTimeoutError:
                status = "timeout"
            except Exception as e:
                status = f"error:{type(e).__name__}"

            # Write output
            with out.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)

                if status == "no_match_in_category":
                    w.writerow([
                        sku, brand, itemname, subcat, source,
                        url, "", "", "", "", "", "", "", "", "", status
                    ])
                else:
                    for hibuddy_url, mscore, mtitle, mrank, mstore_count, rows in match_outputs:
                        if rows:
                            for rr in rows:
                                w.writerow([
                                    sku, brand, itemname, subcat, source,
                                    url, hibuddy_url, (mtitle or ""),
                                    mrank, f"{mscore:.2f}",
                                    mstore_count if mstore_count is not None else "",
                                    rr.get("_variant_class",""),
                                    rr.get("store_name",""),
                                    rr.get("price",""),
                                    rr.get("row_text",""),
                                    status,
                                ])
                        else:
                            w.writerow([
                                sku, brand, itemname, subcat, source,
                                url, hibuddy_url, (mtitle or ""),
                                mrank, f"{mscore:.2f}",
                                mstore_count if mstore_count is not None else "",
                                "", "", "", "", status
                            ])

            sleep(args.min_delay, args.max_delay)

        ctx.close()

def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input1", default="", help="Optional input file #1")
    ap.add_argument("--input2", default="", help="Optional input file #2 (OrderExport etc.)")
    ap.add_argument("--output", default="hibuddy_competition_prices.csv")
    ap.add_argument("--profile-dir", default="hibuddy_profile")
    ap.add_argument("--headless", action="store_true")

    ap.add_argument("--interactive-setup", action="store_true")
    ap.add_argument("--city", default="Toronto")
    ap.add_argument("--province", default="Ontario")
    ap.add_argument("--country", default="Canada")
    ap.add_argument("--radius-km", type=int, default=15)

    ap.add_argument("--resume", action="store_true", help="Resume: skip SKUs already in output")
    ap.add_argument("--cache-category-products", action="store_true", default=True,
                    help="Reuse loaded product links for same category URL")

    ap.add_argument("--max-scrolls", type=int, default=90)
    ap.add_argument("--verify-top-k", type=int, default=3)
    ap.add_argument("--min-match-score", type=float, default=38.0)
    ap.add_argument("--allow-fallback-below-threshold", action="store_true", default=False,
                    help="If no matches pass min-match-score, still keep top candidates (NOT recommended). Default: OFF.")
    ap.add_argument("--max-matches-per-sku", type=int, default=5)
    ap.add_argument("--min-delay", type=float, default=1.4)
    ap.add_argument("--max-delay", type=float, default=2.6)
    return ap.parse_args(argv)

if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
