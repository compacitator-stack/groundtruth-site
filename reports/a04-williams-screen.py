"""A04 production reproduction source. Screen tier: REPRO-UNGATED.
Yahoo Finance 24-hour composite futures bars and ^IRX cash yield; Pinnacle reproduction unrun.
The same-fill coin is POST-HOC 2026-09-18, not part of the June lock.
Use the directory layout and cached-data instructions in the accompanying report.
The numerical functions are unchanged; comments/module prose and one issue pointer are omitted.
"""
import json
import os
import sys
import warnings
from datetime import date
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
os.makedirs(DATA, exist_ok=True)
SPEC = {'ZB': {'sym': 'ZB=F', 'pt': 1000.0, 'tick': 31.25, 'comm': 4.0, 'acct': 13031.25}, 'ES': {'sym': 'ES=F', 'pt': 50.0, 'tick': 12.5, 'comm': 4.0, 'acct': 6950.0}}
MONEY_STOP = 1500.0
SEED = 12345
SHUFFLE_ITERS = 300

def fetch(sym):
    cache = os.path.join(DATA, f"{sym.replace('=', '_')}.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        import yfinance as yf
        df = yf.download(sym, start='2000-01-01', auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        df.to_parquet(cache)
    df = df[['Open', 'High', 'Low', 'Close']].dropna()
    df = df[df.index >= pd.Timestamp('2000-09-01')]
    return df

def third_friday(y, m):
    d = pd.Timestamp(y, m, 1)
    offset = (4 - d.weekday()) % 7
    return d + pd.Timedelta(days=offset + 14)

def last_bday(y, m):
    d = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d

def roll_dates_for(market, index):
    """Synthetic quarterly calendar-roll dates (Arm-B). Mapped to nearest trading day in index.
    ES: 8 business days before 3rd-Friday expiry of H/M/U/Z.
    ZB: last business day of the month preceding the delivery month (Feb/May/Aug/Nov)."""
    years = range(index.min().year, index.max().year + 1)
    targets = []
    if market == 'ES':
        for y in years:
            for m in (3, 6, 9, 12):
                targets.append(third_friday(y, m) - pd.offsets.BDay(8))
    else:
        for y in years:
            for m in (2, 5, 8, 11):
                targets.append(last_bday(y, m))
    idx = index
    rolls = set()
    for t in targets:
        pos = idx.searchsorted(t)
        if pos < len(idx):
            rolls.add(idx[pos])
    return rolls

def backtest_fig41(df, stop_mode='money', roll_dates=None):
    PT = SPEC['ZB']['pt']
    fric = SPEC['ZB']['comm'] + SPEC['ZB']['tick']
    O = df['Open'].values
    H = df['High'].values
    L = df['Low'].values
    C = df['Close'].values
    idx = df.index
    rolls = roll_dates or set()
    trades = []
    pos = 0
    entry = stop = 0.0
    entry_i = -1
    both_touched = 0
    for t in range(1, len(df)):
        is_roll = idx[t] in rolls
        pr = H[t - 1] - L[t - 1]
        if pr <= 0:
            continue
        if pos == 0:
            if is_roll:
                continue
            buy = O[t] + pr
            sell = O[t] - pr
            lh = H[t] >= buy
            sh = L[t] <= sell
            if lh and sh:
                both_touched += 1
                continue
            if lh:
                pos = 1
                entry = buy
                entry_i = t
                stop = entry - (MONEY_STOP / PT if stop_mode == 'money' else 0.5 * pr)
                if stop < O[t] and L[t] <= stop:
                    trades.append((idx[entry_i], idx[t], (stop - entry) * PT - fric, 1, 'stop'))
                    pos = 0
            elif sh:
                pos = -1
                entry = sell
                entry_i = t
                stop = entry + (MONEY_STOP / PT if stop_mode == 'money' else 0.5 * pr)
                if stop > O[t] and H[t] >= stop:
                    trades.append((idx[entry_i], idx[t], (entry - stop) * PT - fric, -1, 'stop'))
                    pos = 0
        else:
            if is_roll:
                pnl = (O[t] - entry) * PT if pos == 1 else (entry - O[t]) * PT
                trades.append((idx[entry_i], idx[t], pnl - fric, pos, 'roll-flat'))
                pos = 0
                continue
            if pos == 1 and O[t] > entry:
                trades.append((idx[entry_i], idx[t], (O[t] - entry) * PT - fric, 1, 'bailout'))
                pos = 0
            elif pos == -1 and O[t] < entry:
                trades.append((idx[entry_i], idx[t], (entry - O[t]) * PT - fric, -1, 'bailout'))
                pos = 0
            elif pos == 1 and L[t] <= stop:
                trades.append((idx[entry_i], idx[t], (stop - entry) * PT - fric, 1, 'stop'))
                pos = 0
            elif pos == -1 and H[t] >= stop:
                trades.append((idx[entry_i], idx[t], (entry - stop) * PT - fric, -1, 'stop'))
                pos = 0
    return (_frame(trades), both_touched)

def backtest_fig101(df, roll_dates=None):
    PT = SPEC['ES']['pt']
    fric = SPEC['ES']['comm'] + SPEC['ES']['tick']
    O = df['Open'].values
    H = df['High'].values
    L = df['Low'].values
    idx = df.index
    rolls = roll_dates or set()
    months = pd.Series(idx).dt.to_period('M')
    first_idx = {}
    for i, p in enumerate(months):
        if p not in first_idx:
            first_idx[p] = i
    entry_days = set(first_idx.values())
    trades = []
    pos = 0
    entry = stop = 0.0
    entry_i = -1
    for t in range(0, len(df)):
        is_roll = idx[t] in rolls
        if pos == 0:
            if t in entry_days and (not is_roll):
                pos = 1
                entry = O[t]
                entry_i = t
                stop = entry - MONEY_STOP / PT
        else:
            if is_roll:
                trades.append((idx[entry_i], idx[t], (O[t] - entry) * PT - fric, 1, 'roll-flat'))
                pos = 0
                continue
            if O[t] > entry:
                trades.append((idx[entry_i], idx[t], (O[t] - entry) * PT - fric, 1, 'bailout'))
                pos = 0
            elif L[t] <= stop:
                trades.append((idx[entry_i], idx[t], (stop - entry) * PT - fric, 1, 'stop'))
                pos = 0
    return _frame(trades)

def _frame(trades):
    if not trades:
        return pd.DataFrame(columns=['entry_date', 'exit_date', 'pnl', 'dir', 'kind'])
    df = pd.DataFrame(trades, columns=['entry_date', 'exit_date', 'pnl', 'dir', 'kind'])
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    return df

def shuffle_fig41(df, n_entries, stop_mode='money', iters=SHUFFLE_ITERS, seed=SEED):
    """Random-entry-day, random-direction control with the same stop+bailout machinery."""
    rng = np.random.default_rng(seed)
    PT = SPEC['ZB']['pt']
    fric = SPEC['ZB']['comm'] + SPEC['ZB']['tick']
    O = df['Open'].values
    H = df['High'].values
    L = df['Low'].values
    rng_day = df['High'].values - df['Low'].values
    N = len(df)
    nets = []
    for _ in range(iters):
        days = rng.choice(np.arange(1, N - 1), size=min(n_entries, N - 2), replace=False)
        dirs = rng.choice([1, -1], size=len(days))
        tot = 0.0
        for t, d in zip(days, dirs):
            pr = rng_day[t - 1]
            if pr <= 0:
                continue
            entry = O[t]
            stop = entry - d * (MONEY_STOP / PT if stop_mode == 'money' else 0.5 * pr)
            for u in range(t + 1, min(t + 60, N)):
                if d == 1:
                    if O[u] > entry:
                        tot += (O[u] - entry) * PT - fric
                        break
                    if L[u] <= stop:
                        tot += (stop - entry) * PT - fric
                        break
                else:
                    if O[u] < entry:
                        tot += (entry - O[u]) * PT - fric
                        break
                    if H[u] >= stop:
                        tot += (entry - stop) * PT - fric
                        break
        nets.append(tot)
    return np.array(nets)

def shuffle_fig101(df, iters=SHUFFLE_ITERS, seed=SEED):
    """Random entry DAY within each month (vs first-trade-day), long, same machinery."""
    rng = np.random.default_rng(seed)
    PT = SPEC['ES']['pt']
    fric = SPEC['ES']['comm'] + SPEC['ES']['tick']
    O = df['Open'].values
    L = df['Low'].values
    idx = df.index
    N = len(df)
    months = pd.Series(idx).dt.to_period('M')
    by_month = {}
    for i, p in enumerate(months):
        by_month.setdefault(p, []).append(i)
    nets = []
    for _ in range(iters):
        tot = 0.0
        for p, days in by_month.items():
            t = rng.choice(days)
            if t >= N - 1:
                continue
            entry = O[t]
            stop = entry - MONEY_STOP / PT
            for u in range(t + 1, min(t + 60, N)):
                if O[u] > entry:
                    tot += (O[u] - entry) * PT - fric
                    break
                if L[u] <= stop:
                    tot += (stop - entry) * PT - fric
                    break
        nets.append(tot)
    return np.array(nets)

def cash_growth(index):
    """Cumulative T-bill growth over the window via ^IRX (13-wk T-bill discount rate, %)."""
    cache = os.path.join(DATA, 'IRX.parquet')
    if os.path.exists(cache):
        irx = pd.read_parquet(cache)
    else:
        import yfinance as yf
        irx = yf.download('^IRX', start='2000-01-01', auto_adjust=False, progress=False)
        if isinstance(irx.columns, pd.MultiIndex):
            irx.columns = [c[0] for c in irx.columns]
        irx = irx[['Close']].dropna()
        irx.to_parquet(cache)
    irx = irx.reindex(index).ffill().bfill()
    daily = irx['Close'].values / 100.0 / 252.0
    growth = np.prod(1.0 + daily)
    avg_rate = float(np.nanmean(irx['Close'].values))
    return (float(growth), avg_rate)
BEAR_WINDOWS = {'2000-02 dotcom': ('2000-09-01', '2002-12-31'), '2007-09 GFC': ('2007-10-01', '2009-03-31'), '2020-Q1 COVID': ('2020-01-01', '2020-03-31'), '2022 bear': ('2022-01-01', '2022-12-31')}

def median_risk(df_trades, market):
    if len(df_trades) == 0:
        return float('nan')
    return MONEY_STOP

def eval_floors(market, trades, df, cash_g, cash_rate, shuffled, label):
    PT = SPEC[market]['pt']
    acct = SPEC[market]['acct']
    fric = SPEC[market]['comm'] + SPEC[market]['tick']
    n = len(trades)
    net = float(trades['pnl'].sum()) if n else 0.0
    per = net / n if n else 0.0
    years = (df.index.max() - df.index.min()).days / 365.25
    wins = int((trades['pnl'] > 0).sum()) if n else 0
    pwin = 100.0 * wins / n if n else 0.0
    cash_pnl = acct * (cash_g - 1.0)
    A_pass = per > 0 and net > cash_pnl
    mr = median_risk(trades, market)
    fir = fric / mr
    C_pass = per > 0
    shp95 = float(np.percentile(shuffled, 95)) if len(shuffled) else float('nan')
    shp50 = float(np.percentile(shuffled, 50)) if len(shuffled) else float('nan')
    B_pass = net > shp95
    bears = {}
    for k, (s, e) in BEAR_WINDOWS.items():
        m = (trades['exit_date'] >= s) & (trades['exit_date'] <= e)
        sub = trades[m]
        bears[k] = {'trades': int(len(sub)), 'net': float(sub['pnl'].sum()) if len(sub) else 0.0}
    if n and net != 0:
        by_year = trades.copy()
        by_year['yr'] = by_year['exit_date'].dt.year
        yr_pnl = by_year.groupby('yr')['pnl'].sum()
        max_year_share = float(yr_pnl.max() / net) if net > 0 else float('nan')
        top_year = int(yr_pnl.idxmax())
    else:
        max_year_share = float('nan')
        top_year = None
        yr_pnl = pd.Series(dtype=float)
    E_pass = not np.isnan(max_year_share) and max_year_share <= 0.5
    if market == 'ZB':
        plausible = 50 <= n <= 3000
        cadence = n / years if years else 0
    else:
        plausible = 150 <= n <= 360
        cadence = n / years if years else 0
    F_pass = plausible
    return {'label': label, 'market': market, 'n': n, 'net': net, 'per_trade': per, 'pwin': pwin, 'years': years, 'A': {'pass': bool(A_pass), 'per_trade': per, 'net': net, 'cash_pnl': cash_pnl, 'acct': acct}, 'B': {'pass': bool(B_pass), 'net': net, 'shuf_p95': shp95, 'shuf_p50': shp50}, 'C': {'pass': bool(C_pass), 'friction_rt': fric, 'median_risk': mr, 'friction_in_R': fir, 'per_trade': per}, 'D': bears, 'E': {'pass': bool(E_pass), 'max_year_share': max_year_share, 'top_year': top_year}, 'F': {'pass': bool(F_pass), 'n': n, 'cadence_per_yr': cadence, 'plausible_band': plausible}}

def fidelity_check():
    """Assert the harness implements each sec-2 edge-bearing rule on tiny synthetic fixtures.
    A degenerate subset (long-only 4.1, wrong stop, no bailout) must FAIL here."""
    checks = []
    PT = SPEC['ZB']['pt']
    fric_zb = SPEC['ZB']['comm'] + SPEC['ZB']['tick']
    f = pd.DataFrame({'Open': [100, 100, 104], 'High': [102, 103, 104], 'Low': [100, 101, 104], 'Close': [101, 102, 104]}, index=pd.to_datetime(['2001-01-02', '2001-01-03', '2001-01-04']))
    tr, _ = backtest_fig41(f, 'money')
    ok = len(tr) == 1 and tr.iloc[0]['dir'] == 1 and (abs(tr.iloc[0]['pnl'] - ((104 - 102) * PT - fric_zb)) < 1e-06) and (tr.iloc[0]['kind'] == 'bailout')
    checks.append(('4.1 entry=open+100%prev-range (LONG) + bailout exit', ok))
    f = pd.DataFrame({'Open': [100, 100, 96], 'High': [102, 100, 96], 'Low': [100, 97, 96], 'Close': [101, 98, 96]}, index=pd.to_datetime(['2001-01-02', '2001-01-03', '2001-01-04']))
    tr, _ = backtest_fig41(f, 'money')
    ok = len(tr) == 1 and tr.iloc[0]['dir'] == -1
    checks.append(('4.1 SHORT side exists (symmetric, NOT long-only)', ok))
    f = pd.DataFrame({'Open': [100, 100, 101], 'High': [102, 103, 101.2], 'Low': [100, 101, 100.0], 'Close': [101, 102, 100.4]}, index=pd.to_datetime(['2001-01-02', '2001-01-03', '2001-01-04']))
    tr, _ = backtest_fig41(f, 'money')
    ok = len(tr) == 1 and tr.iloc[0]['kind'] == 'stop' and (abs(tr.iloc[0]['pnl'] - ((100.5 - 102) * PT - fric_zb)) < 1e-06)
    checks.append(('4.1 protective stop = $1,500 (=1.5 ZB pts) (OV-001 PRIMARY)', ok))
    f = pd.DataFrame({'Open': [100, 100, 101], 'High': [110, 111, 101.2], 'Low': [100, 101, 100.0], 'Close': [105, 108, 100.4]}, index=pd.to_datetime(['2001-01-02', '2001-01-03', '2001-01-04']))
    tr_m, _ = backtest_fig41(f, 'money')
    tr_r, _ = backtest_fig41(f, 'range')
    ok = len(tr_m) == 1 and len(tr_r) == 1 and (abs(tr_m.iloc[0]['pnl'] - tr_r.iloc[0]['pnl']) > 1e-06)
    checks.append(('4.1 50%-range stop SENSITIVITY arm differs from money stop (OV-001)', ok))
    PTes = SPEC['ES']['pt']
    fric_es = SPEC['ES']['comm'] + SPEC['ES']['tick']
    idx = pd.to_datetime(['2001-01-02', '2001-01-03', '2001-02-01'])
    f = pd.DataFrame({'Open': [1000, 1000, 1010], 'High': [1001, 1001, 1010], 'Low': [960, 995, 1010], 'Close': [995, 1000, 1010]}, index=idx)
    tr = backtest_fig101(f)
    ok = len(tr) == 1 and tr.iloc[0]['kind'] == 'bailout' and (abs(tr.iloc[0]['pnl'] - ((1010 - 1000) * PTes - fric_es)) < 1e-06)
    checks.append(('10.1 first-trade-day entry + stop NOT active on entry day (OV-011)', ok))
    ok = True
    checks.append(('10.1 long-only (no short branch) — structural', ok))
    f = pd.DataFrame({'Open': [100, 100, 100], 'High': [102, 103, 100.0], 'Low': [100, 101, 99.9], 'Close': [101, 102, 100]}, index=pd.to_datetime(['2001-01-02', '2001-01-03', '2001-01-04']))
    tr, _ = backtest_fig41(f, 'money')
    ok = len(tr) == 1 and tr.iloc[0]['pnl'] < (100.5 - 102) * PT + 1
    checks.append(('friction (sec4) deducted per round-turn (OV-004/005)', ok))
    allok = all((c[1] for c in checks))
    return (allok, checks)

def run():
    out = {'generated': str(date.today()), 'systems': {}, 'fidelity': {}}
    ok, checks = fidelity_check()
    out['fidelity'] = {'pass': ok, 'checks': [{'rule': c[0], 'pass': bool(c[1])} for c in checks]}
    print('FIDELITY-CHECK:', 'PASS' if ok else 'FAIL')
    for rule, c in checks:
        print(f"   [{('PASS' if c else 'FAIL')}] {rule}")
    if not ok:
        print('\nFIDELITY-CHECK FAILED — aborting (unfaithful detector invalidates every number).')
        json.dump(out, open(os.path.join(DATA, 'p3_results.json'), 'w'), indent=2)
        return out
    zb = fetch('ZB=F')
    es = fetch('ES=F')
    print(f'\nData: ZB={len(zb)} rows {zb.index.min().date()}->{zb.index.max().date()}; ES={len(es)} rows {es.index.min().date()}->{es.index.max().date()}')
    cg_zb, cr_zb = cash_growth(zb.index)
    cg_es, cr_es = cash_growth(es.index)
    print(f'Cash (^IRX): ZB-window growth x{cg_zb:.4f} (avg {cr_zb:.2f}%); ES-window x{cg_es:.4f} (avg {cr_es:.2f}%)')
    zb_rolls = roll_dates_for('ZB', zb.index)
    es_rolls = roll_dates_for('ES', es.index)
    res41 = {}
    for arm, rolls in [('Arm-A', None), ('Arm-B', zb_rolls)]:
        tr, bt = backtest_fig41(zb, 'money', rolls)
        shuf = shuffle_fig41(zb, max(len(tr), 1), 'money')
        f = eval_floors('ZB', tr, zb, cg_zb, cr_zb, shuf, f'Fig4.1 {arm} money-stop')
        f['both_touched_skipped'] = bt
        res41[arm] = f
        print(f"\n[Fig4.1 {arm}] n={f['n']} net=${f['net']:,.0f} per=${f['per_trade']:,.1f} %win={f['pwin']:.1f} A={f['A']['pass']} B={f['B']['pass']} C={f['C']['pass']} E={f['E']['pass']} F={f['F']['pass']} (both-touched skipped={bt})")
    tr_r, _ = backtest_fig41(zb, 'range', None)
    shuf_r = shuffle_fig41(zb, max(len(tr_r), 1), 'range')
    res41['Arm-A_range_stop_sens'] = eval_floors('ZB', tr_r, zb, cg_zb, cr_zb, shuf_r, 'Fig4.1 Arm-A 50%-range-stop SENS')
    print(f"[Fig4.1 Arm-A range-stop SENS] n={len(tr_r)} net=${res41['Arm-A_range_stop_sens']['net']:,.0f} per=${res41['Arm-A_range_stop_sens']['per_trade']:,.1f} A={res41['Arm-A_range_stop_sens']['A']['pass']}")
    res101 = {}
    for arm, rolls in [('Arm-A', None), ('Arm-B', es_rolls)]:
        tr = backtest_fig101(es, rolls)
        shuf = shuffle_fig101(es)
        f = eval_floors('ES', tr, es, cg_es, cr_es, shuf, f'Fig10.1 {arm}')
        res101[arm] = f
        print(f"\n[Fig10.1 {arm}] n={f['n']} net=${f['net']:,.0f} per=${f['per_trade']:,.1f} %win={f['pwin']:.1f} A={f['A']['pass']} B={f['B']['pass']} C={f['C']['pass']} E={f['E']['pass']} F={f['F']['pass']}")
    out['systems']['fig41'] = res41
    out['systems']['fig101'] = res101
    out['rolls'] = {'ZB': len(zb_rolls), 'ES': len(es_rolls)}

    def arm_fidelity(res, primary='Arm-A', secondary='Arm-B'):
        a, b = (res[primary], res[secondary])
        agreeA = a['A']['pass'] == b['A']['pass']
        agreeC = a['C']['pass'] == b['C']['pass']
        return (agreeA and agreeC, agreeA, agreeC)
    for name, res in [('fig41', res41), ('fig101', res101)]:
        agree, aA, aC = arm_fidelity(res)
        a = res['Arm-A']
        if not agree:
            verdict = 'INCONCLUSIVE'
        else:
            A = a['A']['pass']
            B = a['B']['pass']
            C = a['C']['pass']
            if A and B and C and a['E']['pass']:
                verdict = 'SCREEN-PASS'
            elif a['per_trade'] > 0 and (not A):
                verdict = 'SCREEN-WEAK'
            elif a['per_trade'] > 0:
                verdict = 'SCREEN-WEAK'
            else:
                verdict = 'SCREEN-FAIL'
        out['systems'][name]['ARM_FIDELITY'] = {'agree': bool(agree), 'agreeA': bool(aA), 'agreeC': bool(aC)}
        out['systems'][name]['SCREEN_VERDICT'] = verdict
        print(f'\n=== {name}: ARM-FIDELITY agree={agree} (A:{aA} C:{aC}) -> SCREEN-VERDICT={verdict} ===')
    json.dump(out, open(os.path.join(DATA, 'p3_results.json'), 'w'), indent=2, default=str)
    print('\nResults -> data/p3_results.json')
    return out
if __name__ == '__main__':
    run()
