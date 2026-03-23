"""
Supertrend Screener + Alpaca Paper Trading Engine
==================================================
Screens 500 stocks from NASDAQ-100 + S&P 500 for Supertrend buy signals,
then places bracket orders (entry + stop loss + take profit) on Alpaca paper.
Logs all signals and orders to Google Sheets ("ranging" / supertrend_signals + supertrend_orders).

Setup:
    pip install alpaca-trade-api yfinance pandas numpy requests gspread google-auth

GitHub Actions secrets required:
    ALPACA_API_KEY       Alpaca paper API key
    ALPACA_SECRET_KEY    Alpaca paper secret key
    ALPACA_BASE_URL      https://paper-api.alpaca.markets
    GSPREAD_SA_KEY_JSON  Full contents of Google service account JSON
"""

import os
import json
import time
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    raise ImportError("Run: pip install alpaca-trade-api")

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    raise ImportError("Run: pip install gspread google-auth")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "YOUR_PAPER_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_PAPER_SECRET_KEY")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")

# Google Sheets
GSHEET_NAME  = "ranging"
SIGNALS_TAB  = "supertrend_signals"
ORDERS_TAB   = "supertrend_orders"

# Strategy
ATR_PERIOD       = 10
MULTIPLIER       = 3.0
VOL_MA_PERIOD    = 20

# Risk management
RISK_PER_TRADE_PCT = 0.01
STOP_LOSS_ATR_MULT = 1.5
TAKE_PROFIT_RATIO  = 2.0
MAX_OPEN_POSITIONS = 10
MIN_STOCK_PRICE    = 5.0
MIN_AVG_VOLUME     = 500_000

LOOKBACK_DAYS = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE SHEETS
# ─────────────────────────────────────────────────────────────────────────────

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SIGNALS_HEADERS = [
    "run_timestamp", "signal_date", "ticker", "close", "atr",
    "supertrend", "stop", "target", "risk_per_share",
]

ORDERS_HEADERS = [
    "run_timestamp", "ticker", "order_id", "qty",
    "entry", "stop", "target", "risk_usd", "status",
]


def _gsheet_client():
    raw = os.environ.get("GSPREAD_SA_KEY_JSON")
    if not raw:
        raise EnvironmentError("GSPREAD_SA_KEY_JSON env var not set")
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    tf.write(raw)
    tf.flush()
    creds = Credentials.from_service_account_file(tf.name, scopes=SCOPES)
    return gspread.authorize(creds)


def _ensure_tab(spreadsheet, tab_name: str, headers: list):
    try:
        ws = spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=tab_name, rows=10000, cols=len(headers))
        ws.append_row(headers, value_input_option="RAW")
        log.info(f"  Created new tab: {tab_name}")
    return ws


def log_signals_to_sheet(signals: list, run_ts: str):
    if not signals:
        log.info("  No signals to log.")
        return
    try:
        gc  = _gsheet_client()
        ss  = gc.open(GSHEET_NAME)
        ws  = _ensure_tab(ss, SIGNALS_TAB, SIGNALS_HEADERS)
        rows = []
        for s in signals:
            stop   = round(s["close"] - STOP_LOSS_ATR_MULT * s["atr"], 2)
            target = round(s["close"] + (s["close"] - stop) * TAKE_PROFIT_RATIO, 2)
            rows.append([
                run_ts,
                s.get("date", ""),
                s["ticker"],
                round(s["close"], 2),
                round(s["atr"], 4),
                round(s["supertrend"], 2),
                stop,
                target,
                round(s["close"] - stop, 2),
            ])
        ws.append_rows(rows, value_input_option="RAW")
        log.info(f"  ✓ {len(rows)} signals → '{GSHEET_NAME}' / {SIGNALS_TAB}")
    except Exception as e:
        log.error(f"  ✗ Failed to log signals to Google Sheets: {e}")


def log_orders_to_sheet(orders: list, run_ts: str):
    if not orders:
        log.info("  No orders to log.")
        return
    try:
        gc  = _gsheet_client()
        ss  = gc.open(GSHEET_NAME)
        ws  = _ensure_tab(ss, ORDERS_TAB, ORDERS_HEADERS)
        rows = []
        for o in orders:
            rows.append([
                run_ts,
                o["ticker"],
                o.get("order_id", ""),
                o["qty"],
                round(o["entry"], 2),
                round(o["stop"], 2),
                round(o["target"], 2),
                round((o["entry"] - o["stop"]) * o["qty"], 2),
                o.get("status", ""),
            ])
        ws.append_rows(rows, value_input_option="RAW")
        log.info(f"  ✓ {len(rows)} orders → '{GSHEET_NAME}' / {ORDERS_TAB}")
    except Exception as e:
        log.error(f"  ✗ Failed to log orders to Google Sheets: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers() -> list:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        log.warning(f"Could not fetch S&P 500: {e}")
        return []


def get_nasdaq100_tickers() -> list:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for col in ["Ticker", "Symbol", "Tick"]:
            for tbl in tables:
                if col in tbl.columns:
                    return tbl[col].tolist()
    except Exception as e:
        log.warning(f"Could not fetch NASDAQ-100: {e}")
    return []


def get_universe(max_tickers: int = 500) -> list:
    log.info("Fetching ticker universe …")
    sp500    = get_sp500_tickers()
    ndq100   = get_nasdaq100_tickers()
    combined = list(dict.fromkeys(sp500 + ndq100))
    if not combined:
        log.warning("Scraping failed — using fallback 30 large-caps")
        combined = [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","BRK-B","JPM","UNH",
            "V","XOM","JNJ","PG","MA","HD","CVX","MRK","ABBV","PEP","COST","ADBE",
            "AVGO","CRM","AMD","QCOM","TXN","NEE","RTX","LIN",
        ]
    universe = combined[:max_tickers]
    log.info(f"Universe: {len(universe)} tickers ({len(sp500)} S&P, {len(ndq100)} NDQ100)")
    return universe


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(tickers: list, lookback_days: int = LOOKBACK_DAYS) -> dict:
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)
    log.info(f"Downloading {len(tickers)} tickers ({start.date()} → {end.date()}) …")

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    result   = {}
    min_bars = ATR_PERIOD * 3

    for ticker in tickers:
        try:
            df = raw.xs(ticker, axis=1, level=1).copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=["close", "high", "low", "volume"])
            if len(df) < min_bars:                      continue
            if df["close"].iloc[-1] < MIN_STOCK_PRICE:  continue
            if df["volume"].mean() < MIN_AVG_VOLUME:     continue
            result[ticker] = df.reset_index()
        except Exception:
            continue

    log.info(f"Usable tickers after filtering: {len(result)}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUPERTREND
# ─────────────────────────────────────────────────────────────────────────────

def compute_supertrend(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    high, low, close = df["high"], df["low"], df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    hl_avg  = (high + low) / 2
    upper   = (hl_avg + MULTIPLIER * df["atr"]).values.copy()
    lower   = (hl_avg - MULTIPLIER * df["atr"]).values.copy()
    close_v = close.values.copy()

    supertrend    = np.zeros(len(df))
    trend         = np.zeros(len(df), dtype=int)
    supertrend[0] = upper[0]
    trend[0]      = -1

    for i in range(1, len(df)):
        upper[i] = upper[i] if upper[i] < upper[i-1] or close_v[i-1] > upper[i-1] else upper[i-1]
        lower[i] = lower[i] if lower[i] > lower[i-1] or close_v[i-1] < lower[i-1] else lower[i-1]
        if trend[i-1] == -1:
            if close_v[i] > upper[i-1]: trend[i] = 1;  supertrend[i] = lower[i]
            else:                        trend[i] = -1; supertrend[i] = upper[i]
        else:
            if close_v[i] < lower[i-1]: trend[i] = -1; supertrend[i] = upper[i]
            else:                        trend[i] = 1;  supertrend[i] = lower[i]

    df["supertrend"] = supertrend
    df["trend"]      = trend
    return df


def get_signal(df: pd.DataFrame) -> Optional[dict]:
    df = compute_supertrend(df)
    df["vol_ok"] = df["volume"] > df["volume"].rolling(VOL_MA_PERIOD).mean()
    last, prev   = df.iloc[-1], df.iloc[-2]
    if last["trend"] == 1 and prev["trend"] == -1 and bool(last["vol_ok"]):
        return {
            "close"      : float(last["close"]),
            "atr"        : float(last["atr"]),
            "supertrend" : float(last["supertrend"]),
            "date"       : str(last.get("Date", last.get("date", ""))),
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. ALPACA
# ─────────────────────────────────────────────────────────────────────────────

def get_alpaca_client():
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")


def get_open_positions(api) -> set:
    return {p.symbol for p in api.list_positions()}


def calculate_position_size(equity: float, entry: float, stop: float) -> int:
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    return max(int((equity * RISK_PER_TRADE_PCT) / risk_per_share), 1)


def place_bracket_order(api, ticker: str, signal: dict, equity: float) -> Optional[dict]:
    entry  = signal["close"]
    atr    = signal["atr"]
    stop   = round(entry - STOP_LOSS_ATR_MULT * atr, 2)
    risk   = entry - stop
    target = round(entry + risk * TAKE_PROFIT_RATIO, 2)
    shares = calculate_position_size(equity, entry, stop)

    if shares == 0 or shares * entry < 1.0:
        log.warning(f"  {ticker}: skipping — position size too small")
        return None

    log.info(f"  ORDER → {ticker}  qty={shares}  entry≈${entry:.2f}  stop=${stop:.2f}  target=${target:.2f}  risk=${risk*shares:.0f}")

    try:
        order = api.submit_order(
            symbol        = ticker,
            qty           = shares,
            side          = "buy",
            type          = "market",
            time_in_force = "day",
            order_class   = "bracket",
            stop_loss     = {"stop_price": str(stop)},
            take_profit   = {"limit_price": str(target)},
        )
        return {
            "ticker"  : ticker,
            "order_id": order.id,
            "qty"     : shares,
            "entry"   : entry,
            "stop"    : stop,
            "target"  : target,
            "status"  : order.status,
        }
    except Exception as e:
        log.error(f"  {ticker}: order failed — {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(dry_run: bool = False, max_new_orders: int = 5):
    run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    log.info("=" * 60)
    log.info("  SUPERTREND SCREENER — ALPACA PAPER TRADING")
    log.info(f"  Run: {run_ts}")
    log.info("=" * 60)

    # ── Alpaca ────────────────────────────────────────────────────
    if not dry_run:
        try:
            api    = get_alpaca_client()
            equity = float(api.get_account().equity)
            log.info(f"Alpaca equity: ${equity:,.2f}")
        except Exception as e:
            log.error(f"Cannot connect to Alpaca: {e}")
            return
    else:
        log.info("DRY RUN — no orders will be placed")
        api    = None
        equity = 100_000

    # ── Data ──────────────────────────────────────────────────────
    tickers = get_universe(max_tickers=500)
    data    = fetch_ohlcv(tickers)
    if not data:
        log.error("No data downloaded. Exiting.")
        return

    # ── Positions / slots ─────────────────────────────────────────
    open_positions  = get_open_positions(api) if (api and not dry_run) else set()
    available_slots = MAX_OPEN_POSITIONS - len(open_positions)
    log.info(f"Open positions: {len(open_positions)}  Available slots: {available_slots}")

    if available_slots <= 0:
        log.info("Max open positions reached. No new orders.")
        return

    # ── Screen ────────────────────────────────────────────────────
    log.info(f"Scanning {len(data)} tickers for Supertrend buy signals …")
    signals_found = []
    for ticker, df in data.items():
        if ticker in open_positions:
            continue
        sig = get_signal(df)
        if sig:
            sig["ticker"] = ticker
            signals_found.append(sig)

    log.info(f"Signals found: {len(signals_found)}")

    if signals_found:
        sig_df = pd.DataFrame(signals_found)[["ticker", "date", "close", "atr", "supertrend"]]
        sig_df["stop"]   = (sig_df["close"] - STOP_LOSS_ATR_MULT * sig_df["atr"]).round(2)
        sig_df["target"] = (sig_df["close"] + (sig_df["close"] - sig_df["stop"]) * TAKE_PROFIT_RATIO).round(2)
        sig_df["close"]  = sig_df["close"].round(2)
        sig_df["atr"]    = sig_df["atr"].round(3)
        print("\n" + sig_df.to_string(index=False) + "\n")

    # ── Log signals → Google Sheets ───────────────────────────────
    log.info("Logging signals to Google Sheets …")
    log_signals_to_sheet(signals_found, run_ts)

    # ── Place orders ──────────────────────────────────────────────
    orders_placed = []
    limit = min(available_slots, max_new_orders, len(signals_found))

    for sig in signals_found[:limit]:
        ticker = sig["ticker"]
        if dry_run:
            log.info(f"  [DRY RUN] Would order {ticker} @ ${sig['close']:.2f}")
        else:
            result = place_bracket_order(api, ticker, sig, equity)
            if result:
                orders_placed.append(result)
            time.sleep(0.3)

    # ── Log orders → Google Sheets ────────────────────────────────
    log.info("Logging orders to Google Sheets …")
    log_orders_to_sheet(orders_placed, run_ts)

    # ── Summary ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"  Signals: {len(signals_found)}  Orders placed: {len(orders_placed)}")
    log.info("=" * 60)

    return {"signals": signals_found, "orders": orders_placed}


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Supertrend screener + Alpaca paper trading")
    parser.add_argument("--dry-run",    action="store_true", help="Print signals only, no orders")
    parser.add_argument("--max-orders", type=int, default=5, help="Max new orders per run (default 5)")
    args = parser.parse_args()
    run_screener(dry_run=args.dry_run, max_new_orders=args.max_orders)
