"""I05c reproducible search: the original two experiment mains with portable I/O.

Requires Python, numpy, pandas, pyarrow. Supply licensed SPY data yourself:
  python i05c-reproduce.py --daily-csv daily.csv --minute-parquet SPY.parquet --out results        # corrected (default)
  python i05c-reproduce.py --daily-csv daily.csv --minute-parquet SPY.parquet --out results --raw  # as first published

CORRECTION 2026-09-29: We chose this repair rule after the original results were known. It is a post hoc correction, not part of the original daily pre-registration. We applied it before each re-run.
By default a fixed price-repair rule (repair() below) runs on the daily rows and on the
one-minute rows before either search; it fixes a few bad prints in the source data that inflated the hourly results
first published on 2026-09-23. --raw skips the repair and reproduces those first-published figures exactly.

Daily CSV: date (YYYY-MM-DD), ticker, o, h, l, c. Minute parquet: t (UTC epoch
milliseconds), o, h, l, c, v. The historical run used Massive.com (formerly
Polygon.io) data; buy-and-hold uses the same closes. Prices exclude dividends.
Only closed trades are compounded; an open terminal position is omitted.
The hourly extension is not a gold-session match or a pre-registered run.
The report carries chronology, correction, assumptions and original protocol.
Outputs include historical 'his' fields copied from the specimen's reported
screen, not measurements of his strategy. Runtime depends on machine and data.
"""
import os, json, types, argparse
from pathlib import Path
import numpy as np
import pandas as pd
RAW = False                     # --raw: skip the 2026-09-29 price repair (reproduces the first-published figures)
REPAIR_THR = 0.10
def repair(df):
    """Price repair. We chose this repair rule after the original results were known. It is a post hoc correction, not part of the original daily pre-registration. We applied it before each re-run.
    Rows are never dropped. Minute rows are repaired in input file order before sorting and hourly aggregation; daily rows are repaired in date order after selecting the requested dates and ticker. (1) Compute close flags together from the original neighbouring closes, excluding the first and last rows: if the absolute value of close / neighbour - 1 exceeds 0.10 for both neighbours, replace both its close and its open with the original previous close. (2) Using the repaired closes, flag opens whose absolute ratio difference from both the previous close and their own close exceeds 0.10, excluding the first row; replace each flagged open with the repaired previous close. (3) Using the resulting open and close, replace a high above 1.10 times their maximum with that maximum, and a low below 0.90 times their minimum with that minimum. (4) Enforce final bounds: high is at least max(open, close), and low is at most min(open, close)."""
    d = df.copy(); THR = REPAIR_THR
    o, h, l, c = (d[x].to_numpy(float).copy() for x in ('o', 'h', 'l', 'c'))
    pc, nc = np.roll(c, 1), np.roll(c, -1)
    badc = (np.abs(c / pc - 1) > THR) & (np.abs(c / nc - 1) > THR)
    badc[[0, -1]] = False
    c[badc] = pc[badc]
    o[badc] = c[badc]
    pc2 = np.roll(c, 1)
    bado = (np.abs(o / pc2 - 1) > THR) & (np.abs(o / c - 1) > THR)
    bado[0] = False
    o[bado] = pc2[bado]
    mx, mn = np.maximum(o, c), np.minimum(o, c)
    nh, nl = h > (1 + THR) * mx, l < (1 - THR) * mn
    h[nh], l[nl] = mx[nh], mn[nl]
    h, l = np.maximum(h, mx), np.minimum(l, mn)
    d['o'], d['h'], d['l'], d['c'] = o, h, l, c
    return d
class CsvDaily:
    def __init__(self,path):self.path=path
    def load_daily(self,start,end,tickers):
        df=pd.read_csv(self.path,dtype={'date':str,'ticker':str})
        required={'date','ticker','o','h','l','c'}
        if not required.issubset(df):raise ValueError('Daily CSV requires '+str(sorted(required)))
        return df[(df.date>=start)&(df.date<=end)&df.ticker.isin(tickers)].copy()
TICKER = 'SPY'
START, END = ('2004-01-02', '2024-12-31')
IS_END = '2014-12-31'
R_MULT = 3.8
RISK_F = 0.02
ATR_N = 14
SEEDS = 128
P_GRID = [0.02, 0.04, 0.06, 0.1]
K_GRID = [1.0, 1.5, 2.0, 2.5]
COST_R = 0.05
SEED_BASE = 20260714

def load():
    df = M.load_daily(START, END, tickers=[TICKER]).sort_values('date').reset_index(drop=True)
    if not RAW:
        df = repair(df)
    o = df['o'].to_numpy(float)
    h = df['h'].to_numpy(float)
    l = df['l'].to_numpy(float)
    c = df['c'].to_numpy(float)
    dates = df['date'].to_numpy().astype(str)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = np.full(len(c), np.nan)
    atr[ATR_N] = tr[1:ATR_N + 1].mean()
    for i in range(ATR_N + 1, len(c)):
        atr[i] = (atr[i - 1] * (ATR_N - 1) + tr[i]) / ATR_N
    return (dates, o, h, l, c, atr)

def mask_for(seed, p, n):
    rng = np.random.default_rng(SEED_BASE + seed * 131 + int(p * 1000))
    return rng.random(n) < p

def backtest(mask, o, h, l, c, atr, k, i0, i1, cost_r=0.0):
    """Walk bars [i0,i1). Signal on bar i => enter long at bar i+1 open. Returns array of R-multiples."""
    Rs = []
    pos = False
    entry_px = stop_px = tgt_px = risk = 0.0
    i = i0
    while i < i1:
        if pos:
            oi, hi, li = (o[i], h[i], l[i])
            r = None
            if oi <= stop_px:
                r = (oi - entry_px) / risk
            elif oi >= tgt_px:
                r = (oi - entry_px) / risk
            elif li <= stop_px:
                r = (stop_px - entry_px) / risk
            elif hi >= tgt_px:
                r = (tgt_px - entry_px) / risk
            if r is not None:
                Rs.append(r - cost_r)
                pos = False
        if not pos and i + 1 < i1 and mask[i] and (not np.isnan(atr[i])):
            risk = k * atr[i]
            if risk > 0:
                entry_px = o[i + 1]
                stop_px = entry_px - risk
                tgt_px = entry_px + R_MULT * risk
                pos = True
                i += 1
                continue
        i += 1
    return np.asarray(Rs, float)

def stats(Rs, risk_f=RISK_F):
    if len(Rs) == 0:
        return dict(net=0.0, pf=float('nan'), wr=float('nan'), maxdd=0.0, n=0)
    eq = np.concatenate([[1.0], np.cumprod(1.0 + risk_f * Rs)])
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1.0).min())
    wins = Rs[Rs > 0]
    losses = Rs[Rs < 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float('inf')
    return dict(net=float((eq[-1] - 1) * 100), pf=pf, wr=float(100 * len(wins) / len(Rs)), maxdd=dd * 100, n=int(len(Rs)))

def daily_main():
    dates, o, h, l, c, atr = load()
    n = len(c)
    is_i1 = int((dates <= IS_END).sum())
    is_bh = float(c[is_i1 - 1] / c[0] - 1) * 100
    oos_bh = float(c[n - 1] / c[is_i1] - 1) * 100
    configs = []
    for seed in range(SEEDS):
        for p in P_GRID:
            m = mask_for(seed, p, n)
            for k in K_GRID:
                Rs = backtest(m, o, h, l, c, atr, k, 0, is_i1, cost_r=0.0)
                configs.append((seed, p, k, stats(Rs), Rs))
    is_nets = np.array([cfg[3]['net'] for cfg in configs])
    order = np.argsort(is_nets)[::-1]
    win = configs[order[0]]
    wseed, wp, wk, wis, wRs = win
    wmask = mask_for(wseed, wp, n)
    wRs_oos = backtest(wmask, o, h, l, c, atr, wk, is_i1, n, cost_r=0.0)
    woos = stats(wRs_oos)
    wis_fric = stats(wRs - COST_R)
    topN = 50
    top_idx = order[:topN]
    top_is = np.array([configs[j][3]['net'] for j in top_idx])
    top_oos = []
    for j in top_idx:
        s, p, k, _, _ = configs[j]
        mm = mask_for(s, p, n)
        top_oos.append(stats(backtest(mm, o, h, l, c, atr, k, is_i1, n))['net'])
    top_oos = np.array(top_oos)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    rho = spearman(top_is, top_oos)
    is_beat = float(np.mean(is_nets > is_bh) * 100)
    samp = order[::8][:256]
    oos_nets_samp = []
    for j in samp:
        s, p, k, _, _ = configs[j]
        mm = mask_for(s, p, n)
        oos_nets_samp.append(stats(backtest(mm, o, h, l, c, atr, k, is_i1, n))['net'])
    oos_nets_samp = np.array(oos_nets_samp)
    oos_beat = float(np.mean(oos_nets_samp > oos_bh) * 100)
    is_nets_fric = np.array([stats(cfg[4] - COST_R)['net'] for cfg in configs])
    out = dict(universe=dict(ticker=TICKER, start=START, end=END, is_end=IS_END, n_bars=n, is_bars=is_i1, oos_bars=n - is_i1, is_bh_pct=round(is_bh, 1), oos_bh_pct=round(oos_bh, 1), n_configs=len(configs)), his=dict(net_pct=1669, maxdd_pct=28, wr_pct=31, trades=456, pf=1.89), winner=dict(seed=wseed, p_entry=wp, k_atr=wk, IS={**{q: round(wis[q], 2) for q in wis}}, IS_alpha_vs_bh=round(wis['net'] - is_bh, 1), IS_with_friction={**{q: round(wis_fric[q], 2) for q in wis_fric}}, OOS={**{q: round(woos[q], 2) for q in woos}}, OOS_alpha_vs_bh=round(woos['net'] - oos_bh, 1)), distribution=dict(is_net_median=round(float(np.median(is_nets)), 1), is_net_p90=round(float(np.percentile(is_nets, 90)), 1), is_net_max=round(float(is_nets.max()), 1), pct_configs_negative_IS=round(float(np.mean(is_nets < 0) * 100), 1), pct_configs_beat_bh_IS=round(is_beat, 1), pct_configs_beat_bh_OOS=round(oos_beat, 1), top50_IS_vs_OOS_spearman=round(rho, 3), top50_mean_IS_net=round(float(top_is.mean()), 1), top50_mean_OOS_net=round(float(top_oos.mean()), 1), median_net_frictionless=round(float(np.median(is_nets)), 1), median_net_with_friction=round(float(np.median(is_nets_fric)), 1)), params=dict(R_MULT=R_MULT, RISK_F=RISK_F, ATR_N=ATR_N, SEEDS=SEEDS, P_GRID=P_GRID, K_GRID=K_GRID, COST_R=COST_R, SEED_BASE=SEED_BASE))
    print('=' * 70)
    print(json.dumps(out, indent=2))
    print('=' * 70)
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)
NS=types.SimpleNamespace(**{k:globals()[k] for k in ['SEEDS','P_GRID','K_GRID','ATR_N','COST_R','mask_for','backtest','stats']})
IS_CUT = pd.Timestamp('2015-01-01')

def load_1h():
    df = pd.read_parquet(MIN_FP)
    if not RAW:
        df = repair(df)
    df['dt'] = pd.to_datetime(df['t'], unit='ms')
    g = df.set_index('dt').sort_index().resample('1h').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'), v=('v', 'sum')).dropna(subset=['o'])
    o = g['o'].to_numpy(float)
    h = g['h'].to_numpy(float)
    l = g['l'].to_numpy(float)
    c = g['c'].to_numpy(float)
    idx = g.index
    N = NS.ATR_N
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = np.full(len(c), np.nan)
    atr[N] = tr[1:N + 1].mean()
    for i in range(N + 1, len(c)):
        atr[i] = (atr[i - 1] * (N - 1) + tr[i]) / N
    return (idx, o, h, l, c, atr)

def hourly_main():
    idx, o, h, l, c, atr = load_1h()
    n = len(c)
    is_i1 = int((idx < IS_CUT).sum())
    is_bh = float(c[is_i1 - 1] / c[0] - 1) * 100
    oos_bh = float(c[n - 1] / c[is_i1] - 1) * 100
    configs = []
    for seed in range(NS.SEEDS):
        for p in NS.P_GRID:
            m = NS.mask_for(seed, p, n)
            for k in NS.K_GRID:
                Rs = NS.backtest(m, o, h, l, c, atr, k, 0, is_i1, cost_r=0.0)
                configs.append((seed, p, k, NS.stats(Rs), Rs))
    is_nets = np.array([cfg[3]['net'] for cfg in configs])
    order = np.argsort(is_nets)[::-1]
    wseed, wp, wk, wis, wRs = configs[order[0]]
    wmask = NS.mask_for(wseed, wp, n)
    woos = NS.stats(NS.backtest(wmask, o, h, l, c, atr, wk, is_i1, n))
    wis_fric = NS.stats(wRs - NS.COST_R)
    top_idx = order[:50]
    top_is = np.array([configs[j][3]['net'] for j in top_idx])
    top_oos = []
    for j in top_idx:
        s, p, k, _, _ = configs[j]
        top_oos.append(NS.stats(NS.backtest(NS.mask_for(s, p, n), o, h, l, c, atr, k, is_i1, n))['net'])
    top_oos = np.array(top_oos)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    is_beat = float(np.mean(is_nets > is_bh) * 100)
    samp = order[::8][:256]
    oos_samp = np.array([NS.stats(NS.backtest(NS.mask_for(configs[j][0], configs[j][1], n), o, h, l, c, atr, configs[j][2], is_i1, n))['net'] for j in samp])
    oos_beat = float(np.mean(oos_samp > oos_bh) * 100)
    is_nets_fric = np.array([NS.stats(cfg[4] - NS.COST_R)['net'] for cfg in configs])
    out = dict(universe=dict(ticker='SPY', timeframe='1H (SPY 1-min resampled)', start=str(idx[0]), end=str(idx[-1]), is_cut=str(IS_CUT.date()), n_bars=n, is_bars=is_i1, oos_bars=n - is_i1, is_bh_pct=round(is_bh, 1), oos_bh_pct=round(oos_bh, 1), n_configs=len(configs)), his=dict(net_pct=1669, maxdd_pct=28, wr_pct=31, trades=456, pf=1.89), winner=dict(seed=int(wseed), p_entry=wp, k_atr=wk, IS={q: round(wis[q], 2) for q in wis}, IS_alpha_vs_bh=round(wis['net'] - is_bh, 1), IS_with_friction={q: round(wis_fric[q], 2) for q in wis_fric}, OOS={q: round(woos[q], 2) for q in woos}, OOS_alpha_vs_bh=round(woos['net'] - oos_bh, 1)), distribution=dict(is_net_median=round(float(np.median(is_nets)), 1), is_net_p90=round(float(np.percentile(is_nets, 90)), 1), is_net_max=round(float(is_nets.max()), 1), pct_configs_negative_IS=round(float(np.mean(is_nets < 0) * 100), 1), pct_configs_beat_bh_IS=round(is_beat, 1), pct_configs_beat_bh_OOS=round(oos_beat, 1), top50_IS_vs_OOS_spearman=round(spearman(top_is, top_oos), 3), top50_mean_IS_net=round(float(top_is.mean()), 1), top50_mean_OOS_net=round(float(top_oos.mean()), 1), median_net_frictionless=round(float(np.median(is_nets)), 1), median_net_with_friction=round(float(np.median(is_nets_fric)), 1)))
    print('=' * 70)
    print(json.dumps(out, indent=2))
    print('=' * 70)
    with open(os.path.join(OUTPUT_DIR, 'results_1h.json'), 'w') as f:
        json.dump(out, f, indent=2)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--daily-csv',required=True);p.add_argument('--minute-parquet',required=True)
    p.add_argument('--out',default='results')
    p.add_argument('--raw',action='store_true',help='skip the 2026-09-29 price repair; reproduces the figures first published on 2026-09-23')
    args=p.parse_args();RAW=args.raw
    OUTPUT_DIR=str(Path(args.out));Path(OUTPUT_DIR).mkdir(parents=True,exist_ok=True)
    M=CsvDaily(args.daily_csv);MIN_FP=args.minute_parquet
    daily_main();hourly_main()
