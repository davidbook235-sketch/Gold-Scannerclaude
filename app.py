"""
Gold Swing Trading Scanner
---------------------------
Streamlit app that scans gold-related tickers using a multi-indicator
technical scoring system (EMA trend, RSI, MACD, Bollinger Bands) and
produces a BUY / SELL / HOLD suggestion with an ATR-based stop-loss
and target for swing trading.

DISCLAIMER: This tool is for educational purposes only. It is NOT
financial advice. Always do your own research and consult a licensed
financial advisor before trading.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Gold Swing Scanner", layout="wide")

# ----------------------------- Indicators -----------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, period=20, std_mult=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ----------------------------- Signal Engine -----------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["RSI14"] = rsi(df["Close"], 14)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(df["Close"])
    df["BB_UP"], df["BB_MID"], df["BB_LOW"] = bollinger(df["Close"])
    df["ATR14"] = atr(df, 14)
    return df


def generate_signal(df: pd.DataFrame) -> dict:
    """Weighted scoring across 4 indicators. Score range: -4 (strong sell) to +4 (strong buy)."""
    last = df.iloc[-1]
    score = 0
    reasons = []

    # 1. Trend: EMA20 vs EMA50
    if last["EMA20"] > last["EMA50"]:
        score += 1
        reasons.append("EMA20 > EMA50 → uptrend (+1)")
    else:
        score -= 1
        reasons.append("EMA20 < EMA50 → downtrend (-1)")

    # 2. RSI
    if last["RSI14"] < 30:
        score += 1
        reasons.append(f"RSI {last['RSI14']:.1f} oversold → bullish reversal chance (+1)")
    elif last["RSI14"] > 70:
        score -= 1
        reasons.append(f"RSI {last['RSI14']:.1f} overbought → pullback risk (-1)")
    else:
        reasons.append(f"RSI {last['RSI14']:.1f} neutral (0)")

    # 3. MACD histogram
    if last["MACD_HIST"] > 0 and last["MACD"] > last["MACD_SIGNAL"]:
        score += 1
        reasons.append("MACD above signal line → bullish momentum (+1)")
    elif last["MACD_HIST"] < 0 and last["MACD"] < last["MACD_SIGNAL"]:
        score -= 1
        reasons.append("MACD below signal line → bearish momentum (-1)")
    else:
        reasons.append("MACD mixed (0)")

    # 4. Bollinger Band position
    if last["Close"] <= last["BB_LOW"]:
        score += 1
        reasons.append("Price at/below lower Bollinger Band → oversold zone (+1)")
    elif last["Close"] >= last["BB_UP"]:
        score -= 1
        reasons.append("Price at/above upper Bollinger Band → overbought zone (-1)")
    else:
        reasons.append("Price within Bollinger Bands (0)")

    if score >= 3:
        signal = "STRONG BUY"
    elif score >= 1:
        signal = "BUY"
    elif score <= -3:
        signal = "STRONG SELL"
    elif score <= -1:
        signal = "SELL"
    else:
        signal = "HOLD"

    price = last["Close"]
    atr_val = last["ATR14"]
    if "BUY" in signal:
        stop = price - 1.5 * atr_val
        target = price + 3 * atr_val
    elif "SELL" in signal:
        stop = price + 1.5 * atr_val
        target = price - 3 * atr_val
    else:
        stop = price - 1.5 * atr_val
        target = price + 1.5 * atr_val

    return {
        "signal": signal,
        "score": score,
        "price": price,
        "rsi": last["RSI14"],
        "stop": stop,
        "target": target,
        "atr": atr_val,
        "reasons": reasons,
    }


@st.cache_data(ttl=900)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Forex/spot tickers (e.g. XAUUSD=X) often have no/NaN Volume data.
    # Only require OHLC to be present; don't drop rows for missing Volume.
    required_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df = df.dropna(subset=required_cols)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)
    return df


def load_data_with_fallback(symbols: list, period: str, interval: str):
    """Try a list of candidate ticker symbols in order; return first that works.
    Some Yahoo Finance forex/spot symbols (e.g. XAUUSD=X) intermittently
    return an empty response, so we fall back to alternates automatically."""
    for sym in symbols:
        try:
            df = load_data(sym, period, interval)
            if not df.empty and len(df) >= 55:
                return df, sym
        except Exception:
            continue
    return pd.DataFrame(), None


GRAMS_PER_TROY_OZ = 31.1034768


@st.cache_data(ttl=900)
def load_mcx_gold_synthetic(period: str, interval: str) -> pd.DataFrame:
    """MCX Gold isn't directly available via Yahoo Finance/yfinance, so we
    build an approximate ₹/10g series from COMEX Gold futures (GC=F, USD/oz)
    converted through the USD-INR exchange rate. This tracks MCX Gold closely
    but excludes import duty, GST and local premium — treat as an
    approximation, not the exact MCX quote."""
    gc = yf.download("GC=F", period=period, interval=interval, progress=False)
    usdinr = yf.download("USDINR=X", period=period, interval=interval, progress=False)
    if isinstance(gc.columns, pd.MultiIndex):
        gc.columns = gc.columns.get_level_values(0)
    if isinstance(usdinr.columns, pd.MultiIndex):
        usdinr.columns = usdinr.columns.get_level_values(0)
    if gc.empty or usdinr.empty:
        return pd.DataFrame()
    gc = gc.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in gc.columns])
    rate = usdinr["Close"].reindex(gc.index).ffill().bfill()
    factor = (10 / GRAMS_PER_TROY_OZ) * rate
    synthetic = pd.DataFrame(index=gc.index)
    for col in ["Open", "High", "Low", "Close"]:
        synthetic[col] = gc[col] * factor
    synthetic["Volume"] = gc["Volume"].fillna(0) if "Volume" in gc.columns else 0
    synthetic = synthetic.dropna(subset=["Open", "High", "Low", "Close"])
    return synthetic


# ----------------------------- UI -----------------------------

st.title("🥇 Gold Swing Trading Scanner")
st.caption(
    "Multi-indicator technical scanner for gold swing trades. "
    "Educational tool only — not financial advice."
)

GOLD_TICKERS = {
    "Gold Futures (GC=F)": "GC=F",
    "Micro Gold Futures (MGC=F)": "MGC=F",
    "SPDR Gold Shares ETF (GLD)": "GLD",
    "iShares Gold Trust (IAU)": "IAU",
    "Gold Spot / USD (XAUUSD=X)": ["XAUUSD=X", "XAU=X", "GC=F"],
    "MCX Gold ₹/10g (approx, via COMEX+USDINR)": "MCX_SYNTHETIC",
}

with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Candle interval (swing = daily/weekly)", ["1d", "1wk"], index=0)
    selected_names = st.multiselect(
        "Tickers to scan", list(GOLD_TICKERS.keys()), default=list(GOLD_TICKERS.keys())
    )
    st.markdown("---")
    st.caption("Scanner logic: EMA20/50 trend + RSI14 + MACD + Bollinger Bands, "
               "combined into one score. ATR14 sets stop-loss & target.")

if not selected_names:
    st.warning("Sidebar se kam se kam ek ticker select karein.")
    st.stop()

# --------- Scanner table ---------
st.subheader("📊 Scanner — All Tickers")
rows = []
data_cache = {}
for name in selected_names:
    ticker = GOLD_TICKERS[name]
    try:
        if ticker == "MCX_SYNTHETIC":
            df = load_mcx_gold_synthetic(period, interval)
        elif isinstance(ticker, list):
            df, used_symbol = load_data_with_fallback(ticker, period, interval)
        else:
            df = load_data(ticker, period, interval)

        if df.empty or len(df) < 55:
            reason = "empty response" if df.empty else f"only {len(df)} rows (need ≥55)"
            rows.append({"Ticker": name, "Signal": f"NO DATA ({reason})", "Score": "-", "Price": "-",
                         "RSI": "-", "Stop": "-", "Target": "-"})
            continue
        df_ind = compute_indicators(df)
        result = generate_signal(df_ind)
        data_cache[name] = (df_ind, result)
        rows.append({
            "Ticker": name,
            "Signal": result["signal"],
            "Score": result["score"],
            "Price": round(result["price"], 2),
            "RSI": round(result["rsi"], 1),
            "Stop": round(result["stop"], 2),
            "Target": round(result["target"], 2),
        })
    except Exception as e:
        rows.append({"Ticker": name, "Signal": f"ERROR: {e}", "Score": "-", "Price": "-",
                     "RSI": "-", "Stop": "-", "Target": "-"})

scan_df = pd.DataFrame(rows)


def color_signal(val):
    if "BUY" in str(val):
        return "background-color: #1e5f3d; color: white"
    if "SELL" in str(val):
        return "background-color: #7a1f1f; color: white"
    if val == "HOLD":
        return "background-color: #555500; color: white"
    return ""


styler = scan_df.style
if hasattr(styler, "map"):
    styler = styler.map(color_signal, subset=["Signal"])
else:
    styler = styler.applymap(color_signal, subset=["Signal"])
st.dataframe(styler, use_container_width=True)

st.markdown("---")

# --------- Detail view ---------
st.subheader("🔍 Detailed Chart & Reasoning")
detail_name = st.selectbox("Ticker for detail view", selected_names)

if detail_name in data_cache:
    df_ind, result = data_cache[detail_name]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal", result["signal"])
    c2.metric("Price", f"{result['price']:.2f}")
    c3.metric("Suggested Stop", f"{result['stop']:.2f}")
    c4.metric("Suggested Target", f"{result['target']:.2f}")

    st.markdown("**Reasoning breakdown:**")
    for r in result["reasons"]:
        st.write(f"- {r}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_ind.index, open=df_ind["Open"], high=df_ind["High"],
        low=df_ind["Low"], close=df_ind["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA20"], name="EMA20",
                              line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA50"], name="EMA50",
                              line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_UP"], name="BB Upper",
                              line=dict(color="gray", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_LOW"], name="BB Lower",
                              line=dict(color="gray", width=1, dash="dot")))
    fig.update_layout(height=550, xaxis_rangeslider_visible=False,
                       title=f"{detail_name} — Price with EMA & Bollinger Bands")
    st.plotly_chart(fig, use_container_width=True)

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI14"], name="RSI14",
                                  line=dict(color="purple")))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
    rsi_fig.update_layout(height=250, title="RSI (14)")
    st.plotly_chart(rsi_fig, use_container_width=True)

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD_SIGNAL"], name="Signal"))
    macd_fig.add_trace(go.Bar(x=df_ind.index, y=df_ind["MACD_HIST"], name="Histogram"))
    macd_fig.update_layout(height=250, title="MACD")
    st.plotly_chart(macd_fig, use_container_width=True)
else:
    st.info("Is ticker ka data load nahi hua — details ke liye upar scanner table check karein.")

st.markdown("---")
st.caption(
    "⚠️ Disclaimer: Yeh tool sirf educational/informational purpose ke liye hai. "
    "Yeh financial advice nahi hai. Trading se pehle apni research karein ya "
    "SEBI-registered financial advisor se consult karein."
    )
        
