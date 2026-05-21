"""
FinSkillsTracer v5.3 — Real Financial APIs + Semantic Skill Router.

Key improvements over v5.2:
  1. Real API tools: Alpha Vantage (fundamentals, quotes, economic data) + SerpAPI (web/news search)
  2. Ticker resolution: built-in S&P 500 map → AV SYMBOL_SEARCH → LLM fallback
  3. Rate limiting + caching for Alpha Vantage free tier (5 calls/min)
  4. Enhanced SSE events with data source tracking (live/cached/error)

Usage:
    cd FinSkillsTracer
    python web/server_5.3.py
    # Opens at http://localhost:5003
"""

import os
import sys
import json
import time
import re
import requests
from typing import Dict, Any, List, Tuple, Generator, Optional, Callable
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from src.models import canonical_tool_name
from src.llm_client import LLMClient, get_client

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
SKILLS: List[Dict] = []
METRICS: Dict = {}
TRACES: List[Dict] = []

# ---------------------------------------------------------------------------
# Company Name → Ticker Map (~250 major companies)
# ---------------------------------------------------------------------------
COMPANY_TICKER_MAP: Dict[str, str] = {
    # US Tech
    "apple": "AAPL", "apple inc": "AAPL",
    "microsoft": "MSFT", "microsoft corp": "MSFT",
    "nvidia": "NVDA", "nvidia corp": "NVDA",
    "google": "GOOGL", "alphabet": "GOOGL", "alphabet inc": "GOOGL",
    "amazon": "AMZN", "amazon.com": "AMZN", "amazon inc": "AMZN",
    "meta": "META", "facebook": "META", "meta platforms": "META",
    "tesla": "TSLA", "tesla inc": "TSLA",
    "netflix": "NFLX", "netflix inc": "NFLX",
    "broadcom": "AVGO", "broadcom inc": "AVGO",
    "amd": "AMD", "advanced micro devices": "AMD",
    "intel": "INTC", "intel corp": "INTC",
    "qualcomm": "QCOM",
    "oracle": "ORCL", "oracle corp": "ORCL",
    "salesforce": "CRM", "salesforce inc": "CRM",
    "adobe": "ADBE", "adobe inc": "ADBE",
    "cisco": "CSCO", "cisco systems": "CSCO",
    "paypal": "PYPL", "paypal holdings": "PYPL",
    "uber": "UBER", "uber technologies": "UBER",
    "airbnb": "ABNB",
    "palantir": "PLTR",
    "snowflake": "SNOW",
    # Financials
    "berkshire hathaway": "BRK.B",
    "jpmorgan": "JPM", "jpmorgan chase": "JPM", "jp morgan": "JPM",
    "bank of america": "BAC",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "wells fargo": "WFC",
    "citigroup": "C", "citi": "C",
    "visa": "V", "visa inc": "V",
    "mastercard": "MA",
    "american express": "AXP",
    "blackrock": "BLK",
    # Healthcare
    "johnson & johnson": "JNJ",
    "unitedhealth": "UNH", "unitedhealth group": "UNH",
    "pfizer": "PFE", "pfizer inc": "PFE",
    "eli lilly": "LLY",
    "moderna": "MRNA",
    "abbvie": "ABBV",
    "merck": "MRK",
    # Consumer / Retail
    "walmart": "WMT", "walmart inc": "WMT",
    "costco": "COST", "costco wholesale": "COST",
    "mcdonald": "MCD", "mcdonalds": "MCD",
    "starbucks": "SBUX",
    "nike": "NKE", "nike inc": "NKE",
    "coca cola": "KO", "coca-cola": "KO",
    "pepsi": "PEP", "pepsico": "PEP",
    "disney": "DIS", "walt disney": "DIS",
    "home depot": "HD",
    # Industrials / Energy
    "exxon": "XOM", "exxon mobil": "XOM",
    "chevron": "CVX",
    "boeing": "BA", "boeing co": "BA",
    "caterpillar": "CAT",
    "generac": "GE",
    # Chinese Companies
    "alibaba": "BABA", "阿里巴巴": "BABA",
    "tencent": "TCEHY", "腾讯": "TCEHY",
    "byd": "BYDDY", "比亚迪": "BYDDY",
    "台积电": "TSM", "tsmc": "TSM",
    "宁德时代": "CATL", "catl": "CATL",
    "baidu": "BIDU", "百度": "BIDU",
    "jd.com": "JD", "京东": "JD",
    "pdd holdings": "PDD", "拼多多": "PDD",
    "nio": "NIO", "蔚来": "NIO",
    "x peng": "XPEV", "小鹏": "XPEV",
    "li auto": "LI", "理想": "LI",
    "netease": "NTES", "网易": "NTES",
    "trip.com": "TCOM",
    "meituan": "MPNGF",
}


# ---------------------------------------------------------------------------
# API Key Loading
# ---------------------------------------------------------------------------
def _load_api_keys() -> Dict[str, Optional[str]]:
    keys = {
        "alphavantage": os.getenv("ALPHAVANTAGE_API_KEY", "").strip('"').strip("'").strip(),
        "serpapi": os.getenv("SERPAPI_KEY", "").strip('"').strip("'").strip(),
    }
    return {k: (v if v else None) for k, v in keys.items()}


API_KEYS = _load_api_keys()


# ---------------------------------------------------------------------------
# APIRequestCache — TTL-based in-memory cache for API responses
# ---------------------------------------------------------------------------
class APIRequestCache:
    def __init__(self):
        self._store: Dict[str, Tuple[float, dict]] = {}

    def get(self, key: str, ttl: int = 3600) -> Optional[dict]:
        if key not in self._store:
            return None
        ts, data = self._store[key]
        if time.time() - ts > ttl:
            del self._store[key]
            return None
        return data

    def set(self, key: str, data: dict):
        self._store[key] = (time.time(), data)

    def clear_expired(self):
        now = time.time()
        expired = [k for k, (ts, _) in self._store.items() if now - ts > 7200]
        for k in expired:
            del self._store[k]

    def stats(self) -> Dict:
        return {"cached_entries": len(self._store)}


# Global cache instance
_CACHE = APIRequestCache()


# ---------------------------------------------------------------------------
# RateLimiter — sliding window per-tool rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self):
        self._windows: Dict[str, List[float]] = {}
        self._limits: Dict[str, int] = {}

    def configure(self, tool_id: str, max_per_minute: int):
        self._limits[tool_id] = max_per_minute

    def can_call(self, tool_id: str) -> bool:
        limit = self._limits.get(tool_id, 100)
        if tool_id not in self._windows:
            return True
        now = time.time()
        self._windows[tool_id] = [t for t in self._windows[tool_id] if now - t < 60]
        return len(self._windows[tool_id]) < limit

    def wait_seconds(self, tool_id: str) -> float:
        if tool_id not in self._windows or not self._windows[tool_id]:
            return 0
        oldest = min(self._windows[tool_id])
        return max(0, 60 - (time.time() - oldest) + 0.5)

    def record_call(self, tool_id: str):
        if tool_id not in self._windows:
            self._windows[tool_id] = []
        self._windows[tool_id].append(time.time())

    def remaining(self, tool_id: str) -> int:
        limit = self._limits.get(tool_id, 100)
        if tool_id not in self._windows:
            return limit
        now = time.time()
        self._windows[tool_id] = [t for t in self._windows[tool_id] if now - t < 60]
        return max(0, limit - len(self._windows[tool_id]))

    def status(self) -> Dict:
        return {
            tid: {"remaining": self.remaining(tid), "limit": self._limits.get(tid, 100)}
            for tid in self._limits
        }


_RATE_LIMITER = RateLimiter()


# ---------------------------------------------------------------------------
# Real Financial Tool Registry (12 tools)
# ---------------------------------------------------------------------------
REAL_TOOL_REGISTRY: Dict[str, Dict] = OrderedDict({
    # === Financial Data (Alpha Vantage) ===
    "get_company_overview": {
        "tool_id": "get_company_overview",
        "name": "Company Overview",
        "category": "fundamentals",
        "api_source": "alphavantage",
        "api_function": "OVERVIEW",
        "input_params": ["symbol"],
        "output_keys": ["Name", "Symbol", "Sector", "Industry", "MarketCapitalization",
                        "PERatio", "EPS", "DividendYield", "52WeekHigh", "52WeekLow",
                        "Description", "Exchange", "Currency"],
        "rate_limit_minute": 5,
        "cache_ttl": 3600,
        "description": "Company profile, sector, market cap, P/E ratio, and business description",
    },
    "get_stock_quote": {
        "tool_id": "get_stock_quote",
        "name": "Stock Quote",
        "category": "market_data",
        "api_source": "alphavantage",
        "api_function": "GLOBAL_QUOTE",
        "input_params": ["symbol"],
        "output_keys": ["price", "change", "change_percent", "volume", "high", "low", "open", "previous_close"],
        "rate_limit_minute": 5,
        "cache_ttl": 60,
        "description": "Real-time stock price, change %, volume, day range",
    },
    "get_income_statement": {
        "tool_id": "get_income_statement",
        "name": "Income Statement",
        "category": "fundamentals",
        "api_source": "alphavantage",
        "api_function": "INCOME_STATEMENT",
        "input_params": ["symbol"],
        "output_keys": ["annualReports"],
        "rate_limit_minute": 5,
        "cache_ttl": 3600,
        "description": "Annual income statements: revenue, gross profit, operating income, net income, EPS",
    },
    "get_balance_sheet": {
        "tool_id": "get_balance_sheet",
        "name": "Balance Sheet",
        "category": "fundamentals",
        "api_source": "alphavantage",
        "api_function": "BALANCE_SHEET",
        "input_params": ["symbol"],
        "output_keys": ["annualReports"],
        "rate_limit_minute": 5,
        "cache_ttl": 3600,
        "description": "Annual balance sheets: total assets, liabilities, equity, debt ratios",
    },
    "get_cash_flow": {
        "tool_id": "get_cash_flow",
        "name": "Cash Flow",
        "category": "fundamentals",
        "api_source": "alphavantage",
        "api_function": "CASH_FLOW",
        "input_params": ["symbol"],
        "output_keys": ["annualReports"],
        "rate_limit_minute": 5,
        "cache_ttl": 3600,
        "description": "Annual cash flow statements: operating, investing, financing cash flows",
    },
    "get_earnings": {
        "tool_id": "get_earnings",
        "name": "Earnings",
        "category": "fundamentals",
        "api_source": "alphavantage",
        "api_function": "EARNINGS",
        "input_params": ["symbol"],
        "output_keys": ["annualEarnings", "quarterlyEarnings"],
        "rate_limit_minute": 5,
        "cache_ttl": 1800,
        "description": "Annual and quarterly earnings with EPS estimates and surprises",
    },
    "get_market_data": {
        "tool_id": "get_market_data",
        "name": "Market & Economic Data",
        "category": "market_data",
        "api_source": "alphavantage",
        "api_function": "variable",
        "input_params": ["indicator"],
        "output_keys": ["name", "interval", "unit", "data"],
        "rate_limit_minute": 5,
        "cache_ttl": 1800,
        "description": "Economic indicators: Treasury yield, CPI, unemployment, GDP, Fed funds rate",
    },
    "get_fx_rate": {
        "tool_id": "get_fx_rate",
        "name": "FX Exchange Rate",
        "category": "fx",
        "api_source": "alphavantage",
        "api_function": "CURRENCY_EXCHANGE_RATE",
        "input_params": ["from_currency", "to_currency"],
        "output_keys": ["exchange_rate", "from_currency", "to_currency", "last_refreshed"],
        "rate_limit_minute": 5,
        "cache_ttl": 300,
        "description": "Real-time currency exchange rate between any two currencies",
    },
    # === Web Search (SerpAPI) ===
    "web_search": {
        "tool_id": "web_search",
        "name": "Web Search",
        "category": "web",
        "api_source": "serpapi",
        "api_function": "google",
        "input_params": ["query", "num_results"],
        "output_keys": ["organic_results", "search_metadata"],
        "rate_limit_minute": 100,
        "cache_ttl": 300,
        "description": "Google organic search results via SerpAPI — titles, snippets, links",
    },
    "search_news": {
        "tool_id": "search_news",
        "name": "Financial News Search",
        "category": "news",
        "api_source": "serpapi",
        "api_function": "google_news",
        "input_params": ["query"],
        "output_keys": ["news_results"],
        "rate_limit_minute": 100,
        "cache_ttl": 300,
        "description": "Google News search via SerpAPI — latest financial news articles",
    },
    # === Web Fetch ===
    "web_fetch": {
        "tool_id": "web_fetch",
        "name": "Web Fetch",
        "category": "web",
        "api_source": "requests",
        "api_function": "fetch",
        "input_params": ["url"],
        "output_keys": ["title", "text", "status_code"],
        "rate_limit_minute": 50,
        "cache_ttl": 600,
        "description": "Fetch and extract text content from any URL",
    },
    # === Comparison ===
    "compare_companies": {
        "tool_id": "compare_companies",
        "name": "Company Comparison",
        "category": "fundamentals",
        "api_source": "alphavantage",
        "api_function": "multi",
        "input_params": ["symbols"],
        "output_keys": ["companies"],
        "rate_limit_minute": 2,
        "cache_ttl": 3600,
        "description": "Side-by-side comparison of multiple companies: market cap, P/E, revenue, growth",
    },
})


# ===========================================================================
# Real Tool Executor Functions
# ===========================================================================

def _exec_alphavantage(endpoint: str, params: Dict, api_key: str) -> Dict:
    """Generic Alpha Vantage API caller."""
    base = "https://www.alphavantage.co/query"
    full_params = {"apikey": api_key, **params}
    if endpoint:
        full_params["function"] = endpoint
    try:
        resp = requests.get(base, params=full_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Alpha Vantage returns error messages in a dict with a single key like "Note" or "Error Message"
        if "Error Message" in data:
            return {"error": data["Error Message"], "source": "alphavantage"}
        if "Note" in data:
            return {"error": data["Note"], "source": "alphavantage", "rate_limited": True}
        if "Information" in data:
            return {"error": data["Information"], "source": "alphavantage"}
        return data
    except requests.exceptions.Timeout:
        return {"error": "Alpha Vantage request timed out (15s)", "source": "alphavantage"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "source": "alphavantage"}


def _exec_serpapi_search(params: Dict, api_key: str) -> Dict:
    """SerpAPI Google Search via direct HTTP API (no serpapi package needed)."""
    try:
        full_params = {"api_key": api_key, "engine": "google", "hl": "en", **params}
        resp = requests.get("https://serpapi.com/search.json", params=full_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return {"error": data.get("error", "SerpAPI error"), "source": "serpapi"}
        organic = data.get("organic_results", [])
        return {
            "organic_results": [
                {"position": r.get("position"), "title": r.get("title"),
                 "snippet": r.get("snippet", ""), "link": r.get("link", "")}
                for r in organic[:5]
            ],
            "total_results": data.get("search_information", {}).get("total_results", len(organic)),
        }
    except requests.exceptions.Timeout:
        return {"error": "SerpAPI request timed out (15s)", "source": "serpapi"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "source": "serpapi"}


def _exec_serpapi_news(params: Dict, api_key: str) -> Dict:
    """SerpAPI Google News search."""
    try:
        full_params = {"api_key": api_key, "engine": "google", "tbm": "nws", "hl": "en", **params}
        resp = requests.get("https://serpapi.com/search.json", params=full_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return {"error": data.get("error", "SerpAPI error"), "source": "serpapi"}
        news = data.get("news_results", [])
        return {
            "news_results": [
                {"position": r.get("position"), "title": r.get("title"),
                 "snippet": r.get("snippet", ""), "link": r.get("link", ""),
                 "source": r.get("source", {}).get("name", "") if isinstance(r.get("source"), dict) else r.get("source", ""),
                 "date": r.get("date", "")}
                for r in news[:5]
            ],
        }
    except requests.exceptions.Timeout:
        return {"error": "SerpAPI request timed out (15s)", "source": "serpapi"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "source": "serpapi"}


def _exec_web_fetch(params: Dict, _api_key: str = None) -> Dict:
    """Fetch and extract text from a URL."""
    url = params.get("url", "")
    if not url.startswith(("http://", "https://")):
        return {"error": f"Invalid URL: {url}", "source": "requests"}
    try:
        headers = {"User-Agent": "FinSkillsTracer/5.3 (financial research bot)"}
        resp = requests.get(url, headers=headers, timeout=12)
        text = resp.text
        # Simple HTML text extraction (avoid bs4 dependency)
        # Remove script/style tags and their content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', text, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else url
        # Strip all HTML tags
        clean = re.sub(r'<[^>]+>', ' ', text)
        # Collapse whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        # Truncate to ~5000 chars
        return {
            "title": title[:200],
            "text": clean[:5000],
            "status_code": resp.status_code,
            "url": url,
        }
    except requests.exceptions.Timeout:
        return {"error": f"Web fetch timed out for {url}", "source": "requests"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "source": "requests"}


# Tool executor dispatch table
REAL_TOOL_EXECUTORS: Dict[str, Callable] = {
    "get_company_overview": lambda params, key: _exec_alphavantage("OVERVIEW", {"symbol": params.get("symbol", "")}, key),
    "get_stock_quote": lambda params, key: _exec_alphavantage("GLOBAL_QUOTE", {"symbol": params.get("symbol", "")}, key),
    "get_income_statement": lambda params, key: _exec_alphavantage("INCOME_STATEMENT", {"symbol": params.get("symbol", "")}, key),
    "get_balance_sheet": lambda params, key: _exec_alphavantage("BALANCE_SHEET", {"symbol": params.get("symbol", "")}, key),
    "get_cash_flow": lambda params, key: _exec_alphavantage("CASH_FLOW", {"symbol": params.get("symbol", "")}, key),
    "get_earnings": lambda params, key: _exec_alphavantage("EARNINGS", {"symbol": params.get("symbol", "")}, key),
    "get_market_data": lambda params, key: _exec_alphavantage(params.get("indicator", "TREASURY_YIELD"), {}, key),
    "get_fx_rate": lambda params, key: _exec_alphavantage("CURRENCY_EXCHANGE_RATE", {
        "from_currency": params.get("from_currency", "USD"),
        "to_currency": params.get("to_currency", "CNY"),
    }, key),
    "web_search": lambda params, key: _exec_serpapi_search({
        "q": params.get("query", ""), "num": params.get("num_results", 5)
    }, key),
    "search_news": lambda params, key: _exec_serpapi_news({
        "q": params.get("query", ""), "num": params.get("num_results", 5)
    }, key),
    "web_fetch": lambda params, key: _exec_web_fetch(params, key),
    "compare_companies": lambda params, key: _exec_compare_companies(params, key),
}


def _exec_compare_companies(params: Dict, api_key: str) -> Dict:
    """Multi-company comparison aggregating OVERVIEW + GLOBAL_QUOTE for each symbol."""
    symbols = params.get("symbols", [])
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbols:
        return {"error": "No symbols provided for comparison", "source": "alphavantage"}
    if len(symbols) > 5:
        symbols = symbols[:5]

    companies = []
    for sym in symbols:
        overview = _exec_alphavantage("OVERVIEW", {"symbol": sym}, api_key)
        quote = _exec_alphavantage("GLOBAL_QUOTE", {"symbol": sym}, api_key)
        qdata = quote.get("Global Quote", {}) if "Global Quote" in quote else {}
        comp = {
            "symbol": sym,
            "name": overview.get("Name", sym),
            "sector": overview.get("Sector", "N/A"),
            "market_cap": overview.get("MarketCapitalization", "N/A"),
            "pe_ratio": overview.get("PERatio", "N/A"),
            "eps": overview.get("EPS", "N/A"),
            "price": qdata.get("05. price", "N/A"),
            "change_percent": qdata.get("10. change percent", "N/A"),
            "error": overview.get("error") or quote.get("error"),
        }
        companies.append(comp)
    return {"companies": companies, "compared": len(companies)}


# ===========================================================================
# RealToolExecutor — Executes tools via real HTTP APIs
# ===========================================================================

class RealToolExecutor:
    """Executes financial tool sequences via real API calls (Alpha Vantage, SerpAPI).

    Replaces v5.2's MCPToolExecutor which called the LLM to generate fake data.
    Each tool is dispatched to a real HTTP API; LLM is only used for:
      (a) entity extraction from the query
      (b) final answer synthesis from accumulated real data
    """

    def __init__(self, query: str):
        self.query = query
        self.context: List[Dict] = []  # accumulated step results
        self.client = get_client()
        self.errors = 0
        self.errors_list: List[str] = []
        self.data_sources: set = set()

    def execute_sequence(self, tool_sequence: List[str]) -> Generator[Dict, None, None]:
        """Execute tool sequence step by step, yielding SSE events."""
        # Phase: Entity extraction
        yield {"type": "phase", "phase": "extract", "message": "Extracting financial entities from query..."}
        entities = self._extract_entities()
        yield {"type": "entities", "entities": entities, "query_type": entities.get("query_type", "general")}

        # Phase: Tool execution
        yield {"type": "phase", "phase": "execute", "message": f"Executing {len(tool_sequence)} tool(s)..."}

        for i, tool_name in enumerate(tool_sequence):
            canonical = canonical_tool_name(tool_name)
            tool_info = REAL_TOOL_REGISTRY.get(canonical, REAL_TOOL_REGISTRY.get(tool_name))
            if not tool_info:
                yield {
                    "type": "tool_done", "index": i, "tool": canonical,
                    "is_error": True, "error_message": f"Unknown tool: {canonical}",
                    "data_source": "error",
                }
                self.errors += 1
                self.errors_list.append(f"Unknown tool: {canonical}")
                continue

            tool_input = self._build_input(canonical, entities)

            # Check cache
            cache_key = f"{canonical}:{json.dumps(tool_input, sort_keys=True)}"
            cached = _CACHE.get(cache_key, tool_info.get("cache_ttl", 3600))
            data_source = "live"

            # Yield tool_start
            yield {
                "type": "tool_start",
                "index": i,
                "tool": canonical,
                "tool_label": tool_info.get("name", canonical),
                "category": tool_info.get("category", "unknown"),
                "api_source": tool_info.get("api_source", "unknown"),
                "api_function": tool_info.get("api_function", ""),
                "input_params": tool_input,
                "estimated_ms": tool_info.get("rate_limit_minute", 5) * 200,
                "cached": cached is not None,
            }

            start_ms = time.time() * 1000

            rate_limit_bucket = tool_info.get("api_source", canonical)  # shared bucket for same API

            if cached is not None:
                output = cached
                data_source = "cached"
                note = f"Using cached data ({tool_info.get('cache_ttl', 3600)}s TTL)"
            elif not _RATE_LIMITER.can_call(rate_limit_bucket):
                wait = _RATE_LIMITER.wait_seconds(rate_limit_bucket)
                cached_fallback = _CACHE.get(cache_key, 99999)
                if cached_fallback:
                    output = cached_fallback
                    data_source = "cached"
                    note = f"Rate limited. Using cached data."
                else:
                    output = {"error": f"Rate limited ({_RATE_LIMITER._limits.get(rate_limit_bucket, 5)}/min). Retry in {wait:.0f}s.", "rate_limited": True}
                    data_source = "error"
                    note = f"API rate limit reached. Retry in {wait:.0f}s."
                    self.errors += 1
                    self.errors_list.append(f"{canonical}: rate limited")
            else:
                try:
                    api_key = API_KEYS.get(tool_info.get("api_source", ""))
                    if not api_key and tool_info.get("api_source") in ("alphavantage", "serpapi"):
                        output = {"error": f"No API key configured for {tool_info['api_source']}", "source": "config"}
                        data_source = "error"
                        note = f"Missing {tool_info['api_source']} API key. Check .env."
                        self.errors += 1
                    else:
                        executor = REAL_TOOL_EXECUTORS.get(canonical)
                        if executor:
                            output = executor(tool_input, api_key)
                        else:
                            output = {"error": f"No executor for {canonical}", "source": "internal"}
                            data_source = "error"
                        if output.get("error"):
                            data_source = "error"
                            self.errors += 1
                            self.errors_list.append(f"{canonical}: {output.get('error', 'unknown')[:100]}")
                        else:
                            _CACHE.set(cache_key, output)
                            _RATE_LIMITER.record_call(rate_limit_bucket)
                            if tool_info["api_source"] not in ("requests",):
                                self.data_sources.add(tool_info["api_source"])
                        note = None
                except Exception as e:
                    output = {"error": str(e), "source": "exception"}
                    data_source = "error"
                    note = f"Exception: {str(e)[:100]}"
                    self.errors += 1
                    self.errors_list.append(f"{canonical}: {str(e)[:100]}")

            elapsed_ms = int(time.time() * 1000 - start_ms)

            step_result = {
                "index": i, "tool": canonical, "input": tool_input,
                "output": output, "duration_ms": elapsed_ms,
                "is_error": output.get("error") is not None,
                "data_source": data_source,
            }
            self.context.append(step_result)

            yield {
                "type": "tool_done",
                "index": i,
                "tool": canonical,
                "tool_label": tool_info.get("name", canonical),
                "input": tool_input,
                "output": output,
                "duration_ms": elapsed_ms,
                "is_error": output.get("error") is not None,
                "error_message": output.get("error", "") if output.get("error") else None,
                "data_source": data_source,
                "note": note if data_source != "live" else None,
                "api_source": tool_info.get("api_source", ""),
            }

            # Space out Alpha Vantage calls to stay under 5/min free tier limit
            api_src = tool_info.get("api_source", "")
            if api_src == "alphavantage" and data_source == "live" and i + 1 < len(tool_sequence):
                next_tool = canonical_tool_name(tool_sequence[i + 1])
                next_info = REAL_TOOL_REGISTRY.get(next_tool, REAL_TOOL_REGISTRY.get(tool_sequence[i + 1], {}))
                if next_info.get("api_source") == "alphavantage":
                    remaining = _RATE_LIMITER.remaining("alphavantage")
                    if remaining <= 1:
                        yield {"type": "phase", "phase": "throttle",
                               "message": f"Alpha Vantage rate limit ({_RATE_LIMITER._limits.get('alphavantage', 5)}/min). Spacing calls..."}
                    time.sleep(1.5)

        # Phase: Answer synthesis
        yield {"type": "phase", "phase": "synthesize", "message": "Synthesizing answer from real financial data..."}
        answer, confidence = self._synthesize_answer()
        total_duration = sum(c.get("duration_ms", 0) for c in self.context)
        yield {
            "type": "answer",
            "answer": answer,
            "confidence": confidence,
            "total_tools": len(tool_sequence),
            "total_duration_ms": total_duration,
            "data_sources": sorted(self.data_sources),
            "errors": self.errors,
            "errors_list": self.errors_list if self.errors else [],
        }

    # ------------------------------------------------------------------
    # Entity Extraction
    # ------------------------------------------------------------------
    def _extract_entities(self) -> Dict[str, Any]:
        if not self.client.available:
            return self._regex_extract_entities()

        prompt = f"""Extract financial entities from this query. Return JSON.

Query: {self.query}

Return:
{{"companies": [{{"name": "...", "ticker": "..."}}],
 "metrics": ["revenue", "eps", "net income", ...],
 "periods": ["FY2024", "Q3 2025", ...],
 "comparison_currency": "USD",
 "query_type": "single_company | multi_company | comparison | macro_economic | general"}}

For Chinese company names, provide the English ticker if known (e.g., 宁德时代 -> CATL, 比亚迪 -> BYDDY).
If unsure about ticker, use null. Return ONLY valid JSON."""

        result = self.client.chat_json(prompt, system="You are a financial data extraction tool. Output only valid JSON.", temperature=0)
        if result:
            result = self._resolve_tickers(result)
            return result
        return self._regex_extract_entities()

    def _regex_extract_entities(self) -> Dict[str, Any]:
        q = self.query.lower()
        companies = []
        for key, ticker in COMPANY_TICKER_MAP.items():
            if key in q or key.replace(" ", "") in q.replace(" ", ""):
                name = key.title()
                if not any(c["ticker"] == ticker for c in companies):
                    companies.append({"name": name, "ticker": ticker})
        if len(companies) > 10:
            companies = companies[:10]

        metrics = []
        metric_keywords = ["revenue", "eps", "earnings", "net income", "gross profit", "operating income",
                           "ebitda", "growth", "margin", "market cap", "pe ratio", "p/e", "dividend",
                           "cash flow", "debt", "equity", "roe", "roa"]
        for m in metric_keywords:
            if m in q:
                metrics.append(m)

        periods = []
        for p in ["FY2024", "FY2023", "FY2025", "Q1", "Q2", "Q3", "Q4"]:
            if p.lower() in q:
                periods.append(p)

        qt = "general"
        if companies and len(companies) == 1:
            qt = "single_company"
        elif len(companies) >= 2:
            qt = "multi_company"

        return {"companies": companies, "metrics": metrics, "periods": periods, "query_type": qt}

    def _resolve_tickers(self, entities: Dict) -> Dict:
        """Fill in missing tickers using built-in map → AV SYMBOL_SEARCH → LLM fallback."""
        for company in entities.get("companies", []):
            if company.get("ticker"):
                continue
            name = company.get("name", "").lower().strip()
            # Tier 1: built-in map
            if name in COMPANY_TICKER_MAP:
                company["ticker"] = COMPANY_TICKER_MAP[name]
                continue
            # Fuzzy: check if name contains or is contained by map entries
            for map_name, ticker in COMPANY_TICKER_MAP.items():
                if map_name in name or name in map_name:
                    company["ticker"] = ticker
                    break
        return entities

    # ------------------------------------------------------------------
    # Build Input Parameters
    # ------------------------------------------------------------------
    def _build_input(self, tool_name: str, entities: Dict) -> Dict:
        canonical = canonical_tool_name(tool_name)
        companies = entities.get("companies", [])
        metrics = entities.get("metrics", [])
        periods = entities.get("periods", [])

        # Get primary company ticker
        ticker = companies[0].get("ticker") if companies else None
        if not ticker and companies:
            name = companies[0].get("name", "")
            name_lower = name.lower().strip()
            ticker = COMPANY_TICKER_MAP.get(name_lower)
        if not ticker:
            ticker = "AAPL"  # ultimate fallback

        # Check prior context for ticker
        if not ticker:
            for ctx in self.context:
                out = ctx.get("output", {})
                if isinstance(out, dict) and "Symbol" in out:
                    ticker = out["Symbol"]
                    break

        # Alpha Vantage tools
        if canonical in ("get_company_overview", "get_stock_quote", "get_income_statement",
                         "get_balance_sheet", "get_cash_flow", "get_earnings"):
            return {"symbol": ticker or "AAPL"}

        if canonical == "get_market_data":
            indicator = "TREASURY_YIELD"
            m = " ".join(metrics).lower() if metrics else ""
            if "cpi" in m or "inflation" in m or "consumer price" in m:
                indicator = "CPI"
            elif "unemployment" in m or "job" in m:
                indicator = "UNEMPLOYMENT"
            elif "gdp" in m or "economic" in m:
                indicator = "REAL_GDP"
            elif "fed" in m or "interest rate" in m:
                indicator = "FEDERAL_FUNDS_RATE"
            return {"indicator": indicator}

        if canonical == "get_fx_rate":
            return {
                "from_currency": entities.get("comparison_currency", "USD"),
                "to_currency": "CNY",
            }

        # SerpAPI tools
        if canonical == "web_search":
            q = self.query[:300]
            if ticker:
                q = f"{q} {ticker}"
            return {"query": q, "num_results": 5}

        if canonical == "search_news":
            q = "financial news"
            if companies:
                q = f"{companies[0].get('name', '')} {' '.join(metrics[:3])} financial"
            elif metrics:
                q = f"{' '.join(metrics[:3])} financial news"
            return {"query": q[:200], "num_results": 5}

        if canonical == "web_fetch":
            return {"url": f"https://finance.yahoo.com/quote/{ticker or 'AAPL'}"}

        if canonical == "compare_companies":
            syms = [c.get("ticker", c.get("name", "")) for c in companies[:5]]
            return {"symbols": syms if syms else ["AAPL", "MSFT"]}

        return {"query": self.query[:200]}

    # ------------------------------------------------------------------
    # Answer Synthesis
    # ------------------------------------------------------------------
    def _synthesize_answer(self) -> Tuple[str, float]:
        if not self.client.available:
            if not self.context:
                return "Unable to generate answer: LLM unavailable and no tool context.", 0.0
            # Return raw data summary
            parts = []
            for c in self.context:
                if not c.get("is_error"):
                    parts.append(f"[{c['tool']}]: {json.dumps(c.get('output', {}), default=str)[:300]}")
            return "\n".join(parts), 0.3

        # Build structured context from tool outputs
        outputs_parts = []
        for c in self.context:
            out = c.get("output", {})
            if c.get("is_error"):
                outputs_parts.append(f"[{c['tool']}] ERROR: {out.get('error', 'unknown')}")
            else:
                # Extract key financial data points
                summary = self._summarize_output(c["tool"], out)
                outputs_parts.append(f"[{c['tool']}] (source: {c.get('data_source', 'unknown')}):\n{summary}")

        outputs_text = "\n\n".join(outputs_parts)

        prompt = f"""Synthesize a comprehensive financial answer from the real API data below.

User Query: {self.query}

Real Financial Data Retrieved:
{outputs_text}

Instructions:
- Provide a clear, professional answer directly addressing the query
- Include specific numbers, ratios, and trends from the data
- Note which data comes from live APIs vs cache
- Format the answer with markdown: use **bold** for key figures, bullet points for lists
- If the data has gaps, acknowledge them honestly
- For comparisons, present side-by-side metrics

Return JSON:
{{"answer": "the comprehensive markdown-formatted answer", "confidence": <number from 0.0 to 1.0>}}

Return ONLY valid JSON."""

        result = self.client.chat_json(prompt, system="You are a senior financial analyst synthesizing real-time data. Be precise, data-driven, and professional.", temperature=0.1)
        if result and "answer" in result:
            return result.get("answer", "N/A"), float(result.get("confidence", 0.5))

        # Fallback
        return "Data retrieved but unable to synthesize answer.", 0.3

    def _summarize_output(self, tool_name: str, output: Dict) -> str:
        """Extract key financial data points from a tool's raw output."""
        if not output or output.get("error"):
            return str(output.get("error", "No data"))[:200]

        canonical = canonical_tool_name(tool_name)

        if canonical == "get_company_overview":
            return (
                f"  Name: {output.get('Name', 'N/A')}\n"
                f"  Sector: {output.get('Sector', 'N/A')} | Industry: {output.get('Industry', 'N/A')}\n"
                f"  Market Cap: {output.get('MarketCapitalization', 'N/A')}\n"
                f"  P/E Ratio: {output.get('PERatio', 'N/A')} | EPS: {output.get('EPS', 'N/A')}\n"
                f"  Dividend Yield: {output.get('DividendYield', 'N/A')}\n"
                f"  52-Week: {output.get('52WeekLow', 'N/A')} - {output.get('52WeekHigh', 'N/A')}\n"
                f"  Description: {output.get('Description', '')[:300]}"
            )

        if canonical == "get_stock_quote":
            gq = output.get("Global Quote", output)
            return (
                f"  Price: {gq.get('05. price', 'N/A')}\n"
                f"  Change: {gq.get('09. change', 'N/A')} ({gq.get('10. change percent', 'N/A')})\n"
                f"  Day Range: {gq.get('03. low', 'N/A')} - {gq.get('04. high', 'N/A')}\n"
                f"  Volume: {gq.get('06. volume', 'N/A')}"
            )

        if canonical == "get_income_statement":
            reports = output.get("annualReports", [])[:3]
            lines = []
            for r in reports:
                lines.append(
                    f"  FY {r.get('fiscalDateEnding', '?')[:4]}: "
                    f"Revenue={r.get('totalRevenue', 'N/A')}, "
                    f"GrossProfit={r.get('grossProfit', 'N/A')}, "
                    f"OpIncome={r.get('operatingIncome', 'N/A')}, "
                    f"NetIncome={r.get('netIncome', 'N/A')}"
                )
            return "\n".join(lines) if lines else str(output)[:300]

        if canonical == "get_balance_sheet":
            reports = output.get("annualReports", [])[:2]
            lines = []
            for r in reports:
                lines.append(
                    f"  FY {r.get('fiscalDateEnding', '?')[:4]}: "
                    f"TotalAssets={r.get('totalAssets', 'N/A')}, "
                    f"TotalLiabilities={r.get('totalLiabilities', 'N/A')}, "
                    f"Equity={r.get('totalShareholderEquity', 'N/A')}"
                )
            return "\n".join(lines) if lines else str(output)[:300]

        if canonical == "get_earnings":
            ae = output.get("annualEarnings", [])[:3]
            lines = []
            for e in ae:
                lines.append(f"  FY {e.get('fiscalDateEnding', '?')[:4]}: reportedEPS={e.get('reportedEPS', 'N/A')}")
            return "\n".join(lines) if lines else str(output)[:300]

        if canonical == "get_market_data":
            data = output.get("data", [])
            if data:
                first = data[0]
                return f"  {output.get('name', 'Indicator')}: {first.get('value', 'N/A')} (as of {first.get('date', '?')})"
            return str(output)[:300]

        if canonical == "get_fx_rate":
            er = output.get("Realtime Currency Exchange Rate", output)
            return (f"  {er.get('1. From_Currency Code', '?')}/{er.get('3. To_Currency Code', '?')}: "
                    f"{er.get('5. Exchange Rate', 'N/A')} (as of {er.get('6. Last Refreshed', '?')})")

        if canonical in ("web_search",):
            results = output.get("organic_results", output.get("results", []))[:3]
            lines = []
            for r in results:
                lines.append(f"  - {r.get('title', '?')}: {r.get('snippet', '')[:200]}")
            return "\n".join(lines) if lines else str(output)[:300]

        if canonical == "search_news":
            results = output.get("news_results", [])[:3]
            lines = []
            for r in results:
                lines.append(f"  - [{r.get('source', '?')}] {r.get('title', '?')}: {r.get('snippet', '')[:150]}")
            return "\n".join(lines) if lines else str(output)[:300]

        if canonical == "compare_companies":
            comps = output.get("companies", [])
            lines = []
            for c in comps:
                lines.append(f"  {c.get('symbol', '?')} ({c.get('name', '?')}): "
                             f"Price={c.get('price', 'N/A')}, P/E={c.get('pe_ratio', 'N/A')}, "
                             f"MCap={c.get('market_cap', 'N/A')}")
            return "\n".join(lines) if lines else str(output)[:300]

        # Default: return key fields
        return json.dumps(output, default=str)[:400]


# ===========================================================================
# v5.2 FEATURE 1: Semantic Skill Router (kept from v5.2, TOOL_REGISTRY → REAL_TOOL_REGISTRY)
# ===========================================================================

def semantic_route_skills(query: str, limit: int = 12) -> List[Dict]:
    """Route query to skills using LLM-based semantic matching."""
    if not SKILLS:
        return []

    q_lower = query.lower()
    q_words = set(q_lower.split())

    # --- Phase 1: Broad keyword retrieval ---
    financial_terms = {
        'revenue', 'eps', 'earnings', 'income', 'growth', 'margin', 'ebitda',
        'fundamental', 'company', 'series', 'discover', 'financial', 'stock',
        'ticker', 'quarter', 'fiscal', 'annual', 'report', 'balance', 'sheet',
        'cash', 'flow', 'ratio', 'valuation', 'market', 'cap', 'dividend', 'yield',
        'profit', 'net', 'operating', 'compare', 'comparison', 'versus',
        'price', 'quote', 'exchange', 'currency', 'fx', 'news', 'overview',
        'economy', 'gdp', 'inflation', 'cpi', 'treasury', 'unemployment',
    }
    finance_bonus = len(q_words & financial_terms) * 0.3

    broad = []
    for skill in SKILLS:
        seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
        search_text = f"{skill.get('name', '')} {skill.get('description', '')} {' '.join(seq)}".lower()
        text_words = set(search_text.split())
        overlap = len(q_words & text_words)
        score = overlap + finance_bonus
        if score > 0.05:
            broad.append((score, skill))

    broad.sort(key=lambda x: x[0], reverse=True)
    broad = broad[:40]

    # --- Phase 2: LLM semantic ranking ---
    if len(broad) <= 1:
        return [_skill_to_candidate(s, i, "B") for i, (_, s) in enumerate(broad)]

    client = get_client()
    if not client.available:
        return [_skill_to_candidate(s, i, "B") for i, (_, s) in enumerate(broad[:limit])]

    top_for_llm = broad[:min(len(broad), max(limit * 3, 15))]

    candidates_text_parts = []
    for llm_idx, (_, skill) in enumerate(top_for_llm):
        seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
        name = skill.get("name", "Unknown")
        desc = skill.get("description", "")[:120]
        candidates_text_parts.append(
            f"[{llm_idx}] {name}\n    Tools: {' → '.join(seq)}\n    Desc: {desc}"
        )
    candidates_text = "\n".join(candidates_text_parts)

    semantic_prompt = f"""You are a financial AI skill router. Rank ALL of these candidate skills by relevance to the query.

Query: {query}

Candidates:
{candidates_text}

For each candidate, evaluate:
- Does this skill's tool sequence directly answer the query?
- Does the skill handle the specific financial metrics mentioned?
- Would executing this skill produce useful results?
- Prefer multi-step pipelines (Overview → Income → Balance Sheet) over single-tool skills for detailed queries

Return a JSON array of ranked indices (most relevant first, include ALL indices):
{{"ranking": [3, 0, 7, 12, ...], "reasoning": "one sentence explaining top pick"}}

Return ONLY valid JSON."""

    try:
        result = client.chat_json(semantic_prompt, system="You are a financial AI skill router. Output only valid JSON.", temperature=0.1)
    except Exception:
        result = None

    if result and "ranking" in result:
        ranking = result["ranking"]
        reasoning = result.get("reasoning", "")
        candidates = []
        for rank_pos, llm_index in enumerate(ranking):
            if llm_index < len(top_for_llm) and len(candidates) < limit:
                _, skill = top_for_llm[llm_index]
                total = len(ranking)
                tier = "A" if rank_pos < max(total // 3, 1) else ("B" if rank_pos < max(total * 2 // 3, 2) else "C")
                c = _skill_to_candidate(skill, len(candidates), tier)
                c["semantic_rank"] = rank_pos + 1
                c["llm_reasoning"] = reasoning[:200] if rank_pos == 0 else ""
                candidates.append(c)

        ranked_ids = {c["skill_id"] for c in candidates}
        for _, skill in top_for_llm:
            if len(candidates) >= limit:
                break
            if skill.get("skill_id") not in ranked_ids:
                c = _skill_to_candidate(skill, len(candidates), "C")
                c["semantic_rank"] = len(candidates) + 1
                candidates.append(c)
        return candidates

    return [_skill_to_candidate(s, i, "B") for i, (_, s) in enumerate(broad[:limit])]


def _skill_to_candidate(skill: Dict, rank: int, tier: str) -> Dict:
    """Convert a raw skill dict to a router candidate."""
    seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
    canonical_seq = [canonical_tool_name(t) for t in seq]

    tools_detail = []
    total_ms = 0
    for t in canonical_seq:
        info = REAL_TOOL_REGISTRY.get(t, {})
        ms = (info.get("rate_limit_minute", 5) * 200) if info else 500
        total_ms += ms
        tools_detail.append({
            "tool": t,
            "category": info.get("category", "unknown"),
            "api_source": info.get("api_source", "unknown"),
            "estimated_ms": ms,
        })

    consensus = skill.get("consensus", {})
    cons_score = consensus.get("consensus_score", 0) if isinstance(consensus, dict) else 0

    return {
        "rank": rank + 1,
        "tier": tier,
        "skill_id": skill.get("skill_id", ""),
        "name": skill.get("name", "Unknown"),
        "tool_sequence": canonical_seq,
        "original_sequence": seq,
        "tools_detail": tools_detail,
        "support": skill.get("support", 0),
        "consensus_score": round(cons_score, 4),
        "config_count": consensus.get("configuration_count", 0) if isinstance(consensus, dict) else 0,
        "estimated_total_ms": total_ms,
        "estimated_tool_calls": len(canonical_seq),
        "description": skill.get("description", "")[:150],
    }


# ===========================================================================
# Data Loading
# ===========================================================================
def load_data():
    global SKILLS, METRICS, TRACES

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    skills_path = os.path.join(base, "skills_bank_v2.json")
    if os.path.exists(skills_path):
        with open(skills_path, "r", encoding="utf-8") as f:
            SKILLS = json.load(f)
        print(f"[5.3] Loaded {len(SKILLS)} skills")

    for metrics_file in ["phase3_results.json", "phase2_results.json"]:
        metrics_path = os.path.join(base, metrics_file)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                md = json.load(f)
            if "quality_metrics" in md:
                METRICS.update(md["quality_metrics"])
            if "utility_benchmark" in md:
                METRICS["utility_benchmark"] = md["utility_benchmark"]
            print(f"[5.3] Loaded metrics from {metrics_file}")
            break

    # Configure rate limiters — Alpha Vantage uses a SHARED bucket (5 calls/min per API key)
    _RATE_LIMITER.configure("alphavantage", 5)
    _RATE_LIMITER.configure("serpapi", 100)
    _RATE_LIMITER.configure("requests", 50)

    # Validate API keys
    av_ok = bool(API_KEYS.get("alphavantage"))
    sp_ok = bool(API_KEYS.get("serpapi"))
    print(f"[5.3] Alpha Vantage API key: {'OK' if av_ok else 'MISSING'}")
    print(f"[5.3] SerpAPI key: {'OK' if sp_ok else 'MISSING'}")


# ===========================================================================
# API Endpoints
# ===========================================================================

@app.route("/api/health")
def health():
    client = get_client()
    return jsonify({
        "status": "ok",
        "version": "5.3",
        "skills_loaded": len(SKILLS),
        "tools_available": len(REAL_TOOL_REGISTRY),
        "llm_available": client.available,
        "llm_provider": client.provider,
        "llm_model": client.model,
        "llm_label": client.label,
        "available_providers": LLMClient.detect_available(),
        "api_keys": {
            "alphavantage": bool(API_KEYS.get("alphavantage")),
            "serpapi": bool(API_KEYS.get("serpapi")),
        },
        "rate_limits": _RATE_LIMITER.status(),
        "cache_stats": _CACHE.stats(),
    })


@app.route("/api/skills/route", methods=["POST"])
def skills_route_v53():
    """Semantic skill router: LLM-based relevance ranking."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    limit = min(data.get("limit", 12), 20)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    candidates = semantic_route_skills(query, limit=limit)
    best = candidates[0] if candidates else None

    return jsonify({
        "query": query,
        "method": "semantic_llm",
        "total_candidates": len(candidates),
        "best_route": best,
        "candidates": candidates,
        "route_summary": {
            "tier_a_count": sum(1 for c in candidates if c.get("tier") == "A"),
            "tier_b_count": sum(1 for c in candidates if c.get("tier") == "B"),
            "tier_c_count": sum(1 for c in candidates if c.get("tier") == "C"),
            "semantic_ranked": True,
        } if candidates else {},
    })


# Map old v5.2 tool names → v5.3 real tools
OLD_TO_NEW_TOOL_MAP = {
    "WebSearch": "web_search", "google_search": "web_search", "google_search_agent": "web_search",
    "WebFetch": "web_fetch", "web_fetch_agent": "web_fetch",
    "discover_companies": "get_company_overview", "mcp__daloopa__discover_companies": "get_company_overview",
    "discover_company_series": "get_earnings", "mcp__daloopa__discover_company_series": "get_earnings",
    "get_company_fundamentals": "get_income_statement", "mcp__daloopa__get_company_fundamentals": "get_income_statement",
    "get_balance_sheet": "get_balance_sheet", "get_cash_flow": "get_cash_flow",
    "compare_companies": "compare_companies",
}

# Intent-driven pipeline templates for v5.3
INTENT_PIPELINES = {
    "single_company": ["get_company_overview", "get_income_statement", "get_stock_quote"],
    "multi_company": ["compare_companies"],
    "comparison": ["compare_companies"],
    "macro_economic": ["get_market_data", "web_search"],
    "general": ["web_search", "search_news"],
}

def _build_smart_pipeline(query: str, candidates: List[Dict]) -> List[str]:
    """Build optimal v5.3 tool pipeline from semantic router results + entity extraction.

    Priority: LLM intent classification → regex keyword fallback → skill bank mapping → web_search default.
    We skip old skill bank tool names because they reference v5.2's simulated tools, not v5.3's real APIs.
    """
    # Step 1: LLM intent classification (fast, cheap classifier prompt)
    client = get_client()
    if client.available:
        prompt = f"""Classify this financial query. Return JSON.

Query: {query}

Return:
{{"query_type": "single_company | multi_company | comparison | macro_economic | general",
 "has_financial_metrics": true/false,
 "reasoning": "one short sentence"}}

Return ONLY valid JSON."""
        try:
            result = client.chat_json(prompt, system="You are a financial query classifier. Output only valid JSON.", temperature=0)
            if result and result.get("query_type"):
                qtype = result["query_type"]
                if qtype in INTENT_PIPELINES:
                    return INTENT_PIPELINES[qtype]
        except Exception:
            pass

    # Step 2: Regex keyword fallback
    ql = query.lower()
    if any(w in ql for w in ["compare", "vs", "versus", "comparison"]):
        return INTENT_PIPELINES["comparison"]
    if any(w in ql for w in ["revenue", "eps", "earnings", "income", "margin", "profit", "p/e", "pe ratio",
                              "market cap", "balance sheet", "cash flow", "dividend", "roe", "roa",
                              "financial", "fundamental", "quarterly", "annual report"]):
        return INTENT_PIPELINES["single_company"]
    if any(w in ql for w in ["economy", "gdp", "inflation", "cpi", "treasury", "unemployment", "fed",
                              "interest rate", "macro"]):
        return INTENT_PIPELINES["macro_economic"]
    if any(w in ql for w in ["news", "latest", "headline", "update"]):
        return ["search_news"]

    # Step 3: Try skill bank mapping as last resort
    if candidates:
        old_seq = candidates[0].get("tool_sequence", [])
        mapped = []
        for t in old_seq:
            canonical = canonical_tool_name(t)
            if canonical in REAL_TOOL_REGISTRY and canonical != "web_search":
                mapped.append(canonical)
            elif canonical in OLD_TO_NEW_TOOL_MAP:
                new_t = OLD_TO_NEW_TOOL_MAP[canonical]
                if new_t not in mapped and new_t != "web_search":
                    mapped.append(new_t)
        if mapped:
            return mapped

    return ["web_search"]


@app.route("/api/trace/execute")
def trace_execute_v53():
    """SSE endpoint: step-by-step real API tool execution."""
    query = request.args.get("query", "").strip()
    skill_id = request.args.get("skill_id", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Resolve tool sequence
    tool_sequence = None
    if skill_id:
        skill = next((s for s in SKILLS if s.get("skill_id") == skill_id), None)
        if skill:
            seq = skill.get("trigger_pattern", {}).get("tool_sequence", [])
            if seq:
                tool_sequence = [canonical_tool_name(t) for t in seq]

    if not tool_sequence:
        candidates = semantic_route_skills(query, limit=3)
        tool_sequence = _build_smart_pipeline(query, candidates)

    executor = RealToolExecutor(query)

    def generate():
        yield f"data: {json.dumps({'type': 'meta', 'tool_sequence': tool_sequence, 'tool_count': len(tool_sequence), 'query': query, 'model': get_client().model, 'provider': get_client().provider})}\n\n"
        for event in executor.execute_sequence(tool_sequence):
            yield f"data: {json.dumps(event)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.route("/api/tools")
def list_tools():
    """List all available real financial tools."""
    tools = []
    for tool_id, info in REAL_TOOL_REGISTRY.items():
        bucket = info.get("api_source", tool_id)
        tools.append({
            "tool_id": tool_id,
            "name": info.get("name", tool_id),
            "category": info.get("category", "unknown"),
            "api_source": info.get("api_source", "unknown"),
            "input_params": info.get("input_params", []),
            "description": info.get("description", ""),
            "rate_limit_per_minute": _RATE_LIMITER._limits.get(bucket, info.get("rate_limit_minute", 100)),
            "cache_ttl_seconds": info.get("cache_ttl", 3600),
            "remaining_calls": _RATE_LIMITER.remaining(bucket),
        })
    return jsonify({"count": len(tools), "tools": tools})


@app.route("/api/tools/<tool_id>/test")
def test_tool(tool_id):
    """Quick test endpoint for a single tool."""
    tool_info = REAL_TOOL_REGISTRY.get(tool_id)
    if not tool_info:
        return jsonify({"error": f"Unknown tool: {tool_id}"}), 404

    api_key = API_KEYS.get(tool_info.get("api_source", ""))
    executor = REAL_TOOL_EXECUTORS.get(tool_id)
    if not executor:
        return jsonify({"error": f"No executor for: {tool_id}"}), 500

    # Use default test params
    test_params = {"symbol": "AAPL", "query": "Apple financial results"}
    if tool_id == "get_market_data":
        test_params = {"indicator": "CPI"}
    elif tool_id == "get_fx_rate":
        test_params = {"from_currency": "USD", "to_currency": "CNY"}
    elif tool_id == "web_search":
        test_params = {"query": "latest financial news", "num_results": 3}
    elif tool_id == "search_news":
        test_params = {"query": "stock market today", "num_results": 3}
    elif tool_id == "web_fetch":
        test_params = {"url": "https://en.wikipedia.org/wiki/Finance"}
    elif tool_id == "compare_companies":
        test_params = {"symbols": ["AAPL", "MSFT"]}

    try:
        output = executor(test_params, api_key)
        return jsonify({
            "tool_id": tool_id,
            "api_source": tool_info.get("api_source"),
            "test_params": test_params,
            "output": output,
            "success": not output.get("error"),
        })
    except Exception as e:
        return jsonify({"tool_id": tool_id, "error": str(e), "success": False}), 500


@app.route("/api/llm/providers")
def llm_providers():
    provider_id = request.args.get("provider", "").strip()
    if provider_id:
        client = LLMClient(provider=provider_id)
        return jsonify({
            "provider": client.provider,
            "label": client.label,
            "model": client.model,
            "available": client.available,
        })
    return jsonify({
        "current_provider": get_client().provider,
        "current_model": get_client().model,
        "current_label": get_client().label,
        "providers": LLMClient.list_providers(),
        "available": LLMClient.detect_available(),
    })


@app.route("/api/llm/switch", methods=["POST"])
def llm_switch():
    data = request.get_json(silent=True) or {}
    new_provider = data.get("provider", "").strip()
    new_model = data.get("model", "").strip() or None
    if not new_provider:
        return jsonify({"error": "provider is required"}), 400
    if new_provider not in LLMClient.get_provider_configs():
        return jsonify({"error": f"Unknown provider: {new_provider}"}), 400

    client = get_client(provider=new_provider, model=new_model)
    return jsonify({
        "switched": True,
        "provider": client.provider,
        "label": client.label,
        "model": client.model,
        "available": client.available,
        "warning": None if client.available else f"No API key set for {new_provider}",
    })


@app.route("/api/skills", methods=["GET", "POST"])
def skills_list():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        query = data.get("q", data.get("query", ""))
        limit = data.get("limit", 5)
    else:
        query = request.args.get("q", "")
        limit = int(request.args.get("limit", 5))

    if query:
        results = semantic_route_skills(query, limit=limit)
        return jsonify({"query": query, "count": len(results), "skills": results})

    return jsonify({"count": len(SKILLS), "skills": SKILLS[:limit]})


@app.route("/api/skills/<skill_id>")
def skill_detail(skill_id):
    for s in SKILLS:
        if s.get("skill_id") == skill_id:
            return jsonify(s)
    return jsonify({"error": "Skill not found"}), 404


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    load_data()
    print(f"\n[5.3] FinSkillsTracer v5.3 starting...")
    print(f"[5.3] LLM: {get_client().label} ({get_client().model})")
    print(f"[5.3] Tools: {len(REAL_TOOL_REGISTRY)} real financial API tools")
    print(f"[5.3] API keys: AV={'✓' if API_KEYS.get('alphavantage') else '✗'}, SerpAPI={'✓' if API_KEYS.get('serpapi') else '✗'}")
    print(f"[5.3] Server: http://localhost:5003")
    print(f"[5.3] Frontend: open web/v5.3.html in browser\n")
    app.run(host="0.0.0.0", port=5003, debug=True, threaded=True)
