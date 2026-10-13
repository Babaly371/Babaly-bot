import datetime
import concurrent.futures
import time
import os
import re
import json
import requests
import ccxt
import pandas as pd
from flask import Flask

# ==========================================
# 1. API WE MODEL SAZLAMALARI
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# 17 SANY MUGT AI MODELI
OPENROUTER_MODEL_MAP = {
    "DeepSeek-R1": "deepseek/deepseek-r1:free",
    "DeepSeek-V3": "deepseek/deepseek-chat:free",
    "Llama-3.3-70B": "meta-llama/llama-3.3-70b-instruct:free",
    "Qwen-2.5-72B": "qwen/qwen-2.5-72b-instruct:free",
    "Gemini-2.0-Flash": "google/gemini-2.0-flash-exp:free",
    "Mistral-Small-24B": "mistralai/mistral-small-24b-instruct-2501:free",
    "Llama-3.1-405B": "meta-llama/llama-3.1-405b-instruct:free",
    "Gemma-2-9B": "google/gemma-2-9b-it:free",
    "Phi-3-Medium": "microsoft/phi-3-medium-128k-instruct:free",
    "Dolphin-Mixtral-8x22B": "cognitivecomputations/dolphin-mixtral-8x22b:free",
    "Qwen-2.5-Coder-32B": "qwen/qwen-2.5-coder-32b-instruct:free",
    "Hermes-3-Llama-70B": "nousresearch/hermes-3-llama-3.1-70b:free",
    "Mistral-Nemo-12B": "mistralai/mistral-nemo:free",
    "OpenChat-7B": "openchat/openchat-7b:free",
    "Toppy-M-7B": "sphericaldyne/toppy-m-7b:free",
    "Zephyr-7B": "huggingfaceh4/zephyr-7b-beta:free"
}

SCORE_FILE = "ai_scores.json"

def load_ai_scores():
    if os.path.exists(SCORE_FILE):
        try:
            with open(SCORE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {model: 1.0 for model in OPENROUTER_MODEL_MAP.keys()}

def save_ai_scores(scores):
    with open(SCORE_FILE, "w") as f:
        json.dump(scores, f, indent=4)

ai_scores = load_ai_scores()
EXCHANGE = ccxt.bybit({'enableRateLimit': True})

# 20 SASY AKTIW (Kripto we Altyn XAUUSDT)
POPULAR_ASSETS = {
    "BTCUSDT": "BTC/USDT",
    "ETHUSDT": "ETH/USDT",
    "SOLUSDT": "SOL/USDT",
    "XAUUSDT": "XAU/USDT", 
    "XRPUSDT": "XRP/USDT",
    "ADAUSDT": "ADA/USDT",
    "DOGEUSDT": "DOGE/USDT",
    "AVAXUSDT": "AVAX/USDT",
    "LINKUSDT": "LINK/USDT",
    "SUIUSDT": "SUI/USDT",
    "NEARUSDT": "NEAR/USDT",
    "PEPEUSDT": "PEPE/USDT",
    "RENDERUSDT": "RENDER/USDT",
    "FETUSDT": "FET/USDT",
    "APTUSDT": "APT/USDT",
    "MATICUSDT": "POL/USDT",
    "ARBUSDT": "ARB/USDT",
    "INJUSDT": "INJ/USDT",
    "OPUSDT": "OP/USDT",
    "ATOMUSDT": "ATOM/USDT"
}

active_open_trades = {}

# ==========================================
# 2. 27 TEHNIKI METRİK (9 Indikator x 3 Timeframe)
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def get_market_data_27_metrics(symbol):
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        if not ohlcv or len(ohlcv) < 60:
            return None

        closes = pd.Series([x[4] for x in ohlcv])
        highs = pd.Series([x[2] for x in ohlcv])
        lows = pd.Series([x[3] for x in ohlcv])
        volumes = pd.Series([x[5] for x in ohlcv])

        rsi_series = calculate_rsi(closes)
        ema_20_series = closes.ewm(span=20, adjust=False).mean()
        sma_50_series = closes.rolling(window=50).mean()
        macd_series = ema_20_series - closes.ewm(span=26, adjust=False).mean()
        pct_change_series = closes.pct_change() * 100

        idx_now, idx_1h, idx_24h = -1, -2, (-25 if len(closes) >= 25 else 0)

        return {
            "price_now": round(float(closes.iloc[idx_now]), 4),
            "price_1h": round(float(closes.iloc[idx_1h]), 4),
            "price_24h": round(float(closes.iloc[idx_24h]), 4),
            "pct_change_now": round(float(pct_change_series.iloc[idx_now]), 2),
            "pct_change_1h": round(float(pct_change_series.iloc[idx_1h]), 2),
            "pct_change_24h": round(float(pct_change_series.iloc[idx_24h]), 2),
            "rsi_now": round(float(rsi_series.iloc[idx_now]), 2),
            "rsi_1h": round(float(rsi_series.iloc[idx_1h]), 2),
            "rsi_24h": round(float(rsi_series.iloc[idx_24h]), 2),
            "ema20_now": round(float(ema_20_series.iloc[idx_now]), 4),
            "ema20_1h": round(float(ema_20_series.iloc[idx_1h]), 4),
            "ema20_24h": round(float(ema_20_series.iloc[idx_24h]), 4),
            "sma50_now": round(float(sma_50_series.iloc[idx_now]), 4),
            "sma50_1h": round(float(sma_50_series.iloc[idx_1h]), 4),
            "sma50_24h": round(float(sma_50_series.iloc[idx_24h]), 4),
            "macd_now": round(float(macd_series.iloc[idx_now]), 4),
            "macd_1h": round(float(macd_series.iloc[idx_1h]), 4),
            "macd_24h": round(float(macd_series.iloc[idx_24h]), 4),
            "volume_now": round(float(volumes.iloc[idx_now]), 2),
            "volume_1h": round(float(volumes.iloc[idx_1h]), 2),
            "volume_24h": round(float(volumes.iloc[idx_24h]), 2),
            "high_now": round(float(highs.iloc[idx_now]), 4),
            "high_1h": round(float(highs.iloc[idx_1h]), 4),
            "high_24h": round(float(highs.iloc[idx_24h]), 4),
            "low_now": round(float(lows.iloc[idx_now]), 4),
            "low_1h": round(float(lows.iloc[idx_1h]), 4),
            "low_24h": round(float(lows.iloc[idx_24h]), 4),
        }
    except Exception:
        return None

def parse_ai_response(raw_text):
    if not raw_text:
        return "GARAŞ", 1.0, 0.5, 50.0
    text = raw_text.upper()
    direction = "GARAŞ"
    if "SATYN AL" in text or "BUY" in text:
        direction = "SATYN AL"
    elif "SAT" in text or "SELL" in text:
        direction = "SAT"

    tp_match = re.search(r'TP:\s*([0-9.]+)\s*%', text)
    sl_match = re.search(r'SL:\s*([0-9.]+)\s*%', text)
    prob_match = re.search(r'PROB:\s*([0-9.]+)\s*%', text)

    tp_val = float(tp_match.group(1)) if tp_match else 1.0
    sl_val = float(sl_match.group(1)) if sl_match else 0.5
    prob_val = float(prob_match.group(1)) if prob_match else 52.0

    return direction, tp_val, sl_val, prob_val

def ask_ai_model(model_name, prompt):
    try:
        if not OPENROUTER_API_KEY:
            return "GARAŞ", 1.0, 0.5, 50.0
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "27-Metric AI Council"
        }
        payload = {
            "model": OPENROUTER_MODEL_MAP.get(model_name),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=25)
        if res.status_code == 200:
            content = res.json()['choices'][0]['message'].get('content', '')
            return parse_ai_response(content)
        return "GARAŞ", 1.0, 0.5, 50.0
    except:
        return "GARAŞ", 1.0, 0.5, 50.0

# ==========================================
# 3. TRADING LOGIC & SCORING SYSTEM
# ==========================================
def update_ai_scores(trade_won, winning_direction, decisions):
    global ai_scores
    for model_name, (d_action, tp, sl, prob) in decisions.items():
        if trade_won:
            if d_action == winning_direction:
                ai_scores[model_name] = round(ai_scores[model_name] + 0.10, 2)
            elif d_action != "GARAŞ":
                ai_scores[model_name] = round(ai_scores[model_name] - 0.10, 2)
        else:
            if d_action == winning_direction:
                ai_scores[model_name] = round(ai_scores[model_name] - 0.10, 2)
            elif d_action != "GARAŞ":
                ai_scores[model_name] = round(ai_scores[model_name] + 0.10, 2)
        if ai_scores[model_name] < 0.0:
            ai_scores[model_name] = 0.0
    save_ai_scores(ai_scores)

def manage_active_trades():
    global active_open_trades
    closed_assets = []
    for asset_key, trade in list(active_open_trades.items()):
        ccxt_sym = POPULAR_ASSETS[asset_key]
        try:
            ticker = EXCHANGE.fetch_ticker(ccxt_sym)
            current_price = ticker['last']
            entry = trade['entry_price']
            tp_pct = trade['tp'] / 100.0
            sl_pct = trade['sl'] / 100.0
            t_type = trade['type']

            if t_type == "SATYN AL":
                is_won = current_price >= entry * (1 + tp_pct)
                is_lost = current_price <= entry * (1 - sl_pct)
            else:
                is_won = current_price <= entry * (1 - tp_pct)
                is_lost = current_price >= entry * (1 + sl_pct)

            if is_won or is_lost:
                print(f"🏁 SÖWDA ÝAPYLDY: {asset_key} | Netije: {'UTUŞ' if is_won else 'ÝALŇYŞ'}")
                update_ai_scores(is_won, t_type, trade['decisions'])
                closed_assets.append(asset_key)
        except Exception as e:
            print(f"⚠️ Ýalňyşlyk: {e}")

    for asset_key in closed_assets:
        del active_open_trades[asset_key]

def execute_trading_cycle():
    global active_open_trades
    print(f"\n🔄 CIKL BAŞLADY: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Açyk Söwda: {len(active_open_trades)}/3")
    
    manage_active_trades()

    if len(active_open_trades) >= 3:
        return

    for asset_key, ccxt_symbol in POPULAR_ASSETS.items():
        if asset_key in active_open_trades or len(active_open_trades) >= 3:
            continue

        d = get_market_data_27_metrics(ccxt_symbol)
        if d is None:
            continue

        prompt = f"""You are a risk manager. Analyze 27 metrics for {asset_key}:
PRICE: Now={d['price_now']} | 1H={d['price_1h']} | 24H={d['price_24h']}
RSI: Now={d['rsi_now']} | 1H={d['rsi_1h']} | 24H={d['rsi_24h']}
EMA20: Now={d['ema20_now']} | SMA50: {d['sma50_now']} | MACD: {d['macd_now']}
Format EXACTLY as: 'UGUR | TP: X% | SL: Y% | PROB: Z%' ('SATYN AL', 'SAT', or 'GARAŞ')."""

        decisions = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(OPENROUTER_MODEL_MAP)) as executor:
            future_to_model = {executor.submit(ask_ai_model, name, prompt): name for name in OPENROUTER_MODEL_MAP.keys()}
            for f in concurrent.futures.as_completed(future_to_model):
                decisions[future_to_model[f]] = f.result()

        direction_scores = {"SATYN AL": 0.0, "SAT": 0.0, "GARAŞ": 0.0}
        direction_voters = {"SATYN AL": [], "SAT": [], "GARAŞ": []}

        for model_name, (d_action, tp, sl, prob) in decisions.items():
            score = ai_scores.get(model_name, 1.0)
            weight = score if score > 0.30 else 0.01
            direction_scores[d_action] += weight
            direction_voters[d_action].append({"tp": tp, "sl": sl, "prob": prob, "weight": weight})

        winning_direction = max(direction_scores, key=direction_scores.get)
        winning_voters = direction_voters[winning_direction]

        if winning_voters and winning_direction != "GARAŞ":
            total_w = sum(v["weight"] for v in winning_voters)
            final_prob = sum(v["prob"] * v["weight"] for v in winning_voters) / total_w
            avg_tp = sum(v["tp"] * v["weight"] for v in winning_voters) / total_w
            avg_sl = sum(v["sl"] * v["weight"] for v in winning_voters) / total_w
        else:
            final_prob, avg_tp, avg_sl = 50.0, 1.0, 0.5

        if winning_direction in ["SATYN AL", "SAT"] and final_prob >= 52.0:
            print(f"🚀 [SIGNAL] {asset_key} -> {winning_direction} | Ähtimallyk: %{final_prob:.1f} | TP: %{avg_tp:.2f}")
            active_open_trades[asset_key] = {
                "entry_price": d['price_now'],
                "type": winning_direction,
                "tp": avg_tp,
                "sl": avg_sl,
                "decisions": decisions,
                "time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

# ==========================================
# 4. FLASK SERVER
# ==========================================
app = Flask(__name__)

@app.route('/')
def home():
    return f"AI 27-Metric Trading Bot Active! Current Open Trades: {len(active_open_trades)}"

if __name__ == "__main__":
    execute_trading_cycle()
