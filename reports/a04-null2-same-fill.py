"""A04 production reproduction source. Screen tier: REPRO-UNGATED.
Yahoo Finance 24-hour composite futures bars and ^IRX cash yield; Pinnacle reproduction unrun.
The same-fill coin is POST-HOC 2026-09-18, not part of the June lock.
Use the directory layout and cached-data instructions in the accompanying report.
The numerical functions are unchanged; comments/module prose and one issue pointer are omitted.
Internal line-number citations are rewritten as the function names they point to, so every citation resolves in this download.
The prior-probe constants are the 60-iteration development probe's published values, recorded here as a fixed reference; they are reproduced from this file's own tables and are not independently sourceable from this download.
"""
import json, os, sys, time
from datetime import date
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
sys.path.insert(0, LAB)
import williams_screen as ws
T0 = time.time()
ITERS = ws.SHUFFLE_ITERS
SEED = ws.SEED
CAP = 60
PT = ws.SPEC['ZB']['pt']
FRIC = ws.SPEC['ZB']['comm'] + ws.SPEC['ZB']['tick']
MONEY = ws.MONEY_STOP / PT
PROBE_ITERS, PROBE_DAYS = (60, 3000)

def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)
zb = ws.fetch('ZB=F')
O = zb['Open'].values
H = zb['High'].values
L = zb['Low'].values
N = len(zb)
PR = H - L
DAYS = np.arange(1, N - 1)
log(f'data rows {N} {zb.index.min().date()} -> {zb.index.max().date()} | eligible days {len(DAYS)}')

def stop_dist(mode, p):
    return MONEY if mode == 'money' else 0.5 * p

def walk(t, d, entry, stop):
    """Engine/probe walk-forward from t+1: first favourable open (bailout) checked BEFORE the stop
    on each bar; 60-bar cap; returns (pnl, kind) or (nan, 'unresolved')."""
    for u in range(t + 1, min(t + CAP, N)):
        if d == 1:
            if O[u] > entry:
                return ((O[u] - entry) * PT - FRIC, 'bailout')
            if L[u] <= stop:
                return ((stop - entry) * PT - FRIC, 'stop')
        else:
            if O[u] < entry:
                return ((entry - O[u]) * PT - FRIC, 'bailout')
            if H[u] >= stop:
                return ((entry - stop) * PT - FRIC, 'stop')
    return (float('nan'), 'unresolved')

def build_tables(mode):
    at_open = {1: np.full(N, np.nan), -1: np.full(N, np.nan)}
    same_fill = {1: np.full(N, np.nan), -1: np.full(N, np.nan)}
    kinds = {1: np.empty(N, dtype=object), -1: np.empty(N, dtype=object)}
    unresolved_open = unresolved_sf = 0
    for t in DAYS:
        p = PR[t - 1]
        if p <= 0:
            continue
        S = stop_dist(mode, p)
        for d in (1, -1):
            entry = O[t]
            stop = entry - d * S
            pnl, kind = walk(t, d, entry, stop)
            if kind == 'unresolved':
                unresolved_open += 1
            at_open[d][t] = pnl
            level = O[t] + d * p
            touched = H[t] >= level if d == 1 else L[t] <= level
            if not touched:
                continue
            entry = level
            stop = entry - d * S
            if d == 1 and stop < O[t] and (L[t] <= stop):
                same_fill[d][t] = (stop - entry) * PT - FRIC
                kinds[d][t] = 'stop0'
                continue
            if d == -1 and stop > O[t] and (H[t] >= stop):
                same_fill[d][t] = (entry - stop) * PT - FRIC
                kinds[d][t] = 'stop0'
                continue
            pnl, kind = walk(t, d, entry, stop)
            if kind == 'unresolved':
                unresolved_sf += 1
            same_fill[d][t] = pnl
            kinds[d][t] = kind
    return (at_open, same_fill, kinds, unresolved_open, unresolved_sf)
TABLES = {}
for mode in ('money', 'range'):
    TABLES[mode] = build_tables(mode)
    log(f'tables built for {mode} stop: unresolved-in-{CAP}-bars at-open {TABLES[mode][3]} | same-fill {TABLES[mode][4]}')
p_all = PR[DAYS - 1]
ok = p_all > 0
up_t = ok & (H[DAYS] >= O[DAYS] + p_all)
dn_t = ok & (L[DAYS] <= O[DAYS] - p_all)
POOL = {'eligible_days': int(len(DAYS)), 'pr_le_0_skipped': int((~ok).sum()), 'up_touched_days': int(up_t.sum()), 'down_touched_days': int(dn_t.sum()), 'both_touched_days': int((up_t & dn_t).sum()), 'any_touched_days': int((up_t | dn_t).sum()), 'pairs': int(up_t.sum() + dn_t.sum()), 'expected_fills_one_coin_per_day_all_days': float(0.5 * (up_t.sum() + dn_t.sum()))}
log(f'fill pool: {POOL}')

def pctl(a):
    a = np.asarray(a, dtype=float)
    p5, p50, p95 = np.percentile(a, [5, 50, 95])
    return {'p5': float(p5), 'p50': float(p50), 'p95': float(p95), 'mean': float(a.mean()), 'min': float(a.min()), 'max': float(a.max())}

def rank_pct(dist, x):
    dist = np.asarray(dist, dtype=float)
    return {'pct_below': float(100.0 * (dist < x).mean()), 'pct_at_or_below': float(100.0 * (dist <= x).mean()), 'n_iters_below': int((dist < x).sum()), 'n_iters': int(len(dist)), 'inside_p5_p95': bool(np.percentile(dist, 5) <= x <= np.percentile(dist, 95))}
P3 = json.load(open(os.path.join(LAB, 'data', 'p3_results.json')))
REAL_PTR = {'money': '/systems/fig41/Arm-A', 'range': '/systems/fig41/Arm-A_range_stop_sens'}
REAL = {}
for mode in ('money', 'range'):
    tr, both_skipped = ws.backtest_fig41(zb, mode, None)
    j = P3['systems']['fig41']['Arm-A' if mode == 'money' else 'Arm-A_range_stop_sens']
    assert len(tr) == j['n'] and abs(float(tr.pnl.sum()) - j['net']) < 1e-06, 'engine re-run != p3_results.json'
    side = {}
    for name, d in (('long', 1), ('short', -1)):
        s = tr[tr.dir == d]
        side[name] = {'n': int(len(s)), 'net': float(s.pnl.sum()), 'per_trade': float(s.pnl.mean()), 'pwin': float(100 * (s.pnl > 0).mean())}
    REAL[mode] = {'pointer': REAL_PTR[mode], 'n': int(j['n']), 'net': float(j['net']), 'per_trade': float(j['per_trade']), 'pwin': float(j['pwin']), 'both_touched_skipped': int(both_skipped), 'long': side['long'], 'short': side['short'], 'as_run_control_pointer_p50': REAL_PTR[mode] + '/B/shuf_p50', 'as_run_control_pointer_p95': REAL_PTR[mode] + '/B/shuf_p95'}
    log(f"REAL {mode}: n={j['n']} net={j['net']:,.2f} per={j['per_trade']:,.2f} | long n={side['long']['n']} per={side['long']['per_trade']:,.2f} | short n={side['short']['n']} per={side['short']['per_trade']:,.2f}")

def at_open_table_run(mode, n, iters=ITERS, seed=SEED, force=None):
    """Table-driven twin of engine shuffle_fig41 with the IDENTICAL rng call sequence (the same two rng.choice calls, in williams_screen.shuffle_fig41).
    force=+1/-1 forces every direction (the long-only / short-only decomposition)."""
    rng = np.random.default_rng(seed)
    tab = TABLES[mode][0]
    nets = []
    for _ in range(iters):
        days = rng.choice(np.arange(1, N - 1), size=min(n, N - 2), replace=False)
        dirs = rng.choice([1, -1], size=len(days))
        if force is not None:
            dirs = np.full_like(dirs, force)
        v = np.where(dirs == 1, tab[1][days], tab[-1][days])
        nets.append(float(np.nansum(v)))
    return np.array(nets)
AS_RUN = {}
for mode in ('money', 'range'):
    n = REAL[mode]['n']
    eng = ws.shuffle_fig41(zb, n, mode)
    twin = at_open_table_run(mode, n)
    eq = int(np.sum(np.abs(eng - twin) < 0.005))
    j50, j95 = (P3['systems']['fig41']['Arm-A' if mode == 'money' else 'Arm-A_range_stop_sens']['B']['shuf_p50'], P3['systems']['fig41']['Arm-A' if mode == 'money' else 'Arm-A_range_stop_sens']['B']['shuf_p95'])
    p = pctl(eng)
    AS_RUN[mode] = {'n_sampled_entries': n, 'iters': ITERS, 'seed': SEED, 'net': p, 'per_trade': {k: p[k] / n for k in ('p5', 'p50', 'p95', 'mean')}, 'per_trade_denominator': 'n sampled entry days (engine returns nets only; pr<=0 and unresolved-in-60 drops are 0.0-0.22 per iteration)', 'json_p50': float(j50), 'json_p95': float(j95), 'reproduces_json_to_cent': bool(abs(p['p50'] - j50) < 0.005 and abs(p['p95'] - j95) < 0.005), 'table_twin_equal_to_engine_iters': f'{eq} of {ITERS}'}
    assert eq == ITERS, 'table-driven twin diverged from engine shuffle_fig41'
    for name, f in (('long_only', 1), ('short_only', -1)):
        nn = REAL[mode][name.split('_')[0]]['n']
        d = at_open_table_run(mode, nn, force=f)
        q = pctl(d)
        AS_RUN[mode][name] = {'n_sampled_entries': nn, 'net': q, 'per_trade_p50': q['p50'] / nn}
    log(f"AS-RUN at-open {mode}: net p50 {p['p50']:,.2f} (json {j50:,.2f}) p95 {p['p95']:,.2f} (json {j95:,.2f}) | twin==engine {eq} of {ITERS} | per-trade p50 {p['p50'] / n:,.2f} | long-only per {AS_RUN[mode]['long_only']['per_trade_p50']:,.2f} short-only per {AS_RUN[mode]['short_only']['per_trade_p50']:,.2f}")
PRIOR_PROBE = {'money': {'per_trade_p50': -162.14, 'p5': -231.95, 'p95': -95.63, 'fills': 507, 'pwin': 0.65}, 'range': {'per_trade_p50': 50.58, 'p5': 14.83, 'p95': 87.79, 'fills': 507, 'pwin': 0.545}}
PROBE = {}
for mode in ('money', 'range'):
    tab = TABLES[mode][1]
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(PROBE_ITERS):
        days = rng.choice(np.arange(1, N - 1), size=PROBE_DAYS, replace=False)
        dirs = rng.choice([1, -1], size=len(days))
        v = np.where(dirs == 1, tab[1][days], tab[-1][days])
        a = v[~np.isnan(v)]
        out.append((a.mean(), len(a), (a > 0).mean()))
    a = np.array(out)
    r = {'per_trade_p50': float(np.median(a[:, 0])), 'per_trade_p5': float(np.percentile(a[:, 0], 5)), 'per_trade_p95': float(np.percentile(a[:, 0], 95)), 'fills_per_iter': float(a[:, 1].mean()), 'pwin': float(a[:, 2].mean()), 'iters': PROBE_ITERS, 'days_sampled': PROBE_DAYS}
    lz = PRIOR_PROBE[mode]
    r['prior_probe_values'] = lz
    r['matches_prior_probe_to_cent'] = bool(abs(r['per_trade_p50'] - lz['per_trade_p50']) < 0.005 and abs(r['per_trade_p5'] - lz['p5']) < 0.005 and (abs(r['per_trade_p95'] - lz['p95']) < 0.005))
    PROBE[mode] = r
    log(f"PROBE replication {mode}: per-trade p50 {r['per_trade_p50']:.2f} (p5 {r['per_trade_p5']:.2f}, p95 {r['per_trade_p95']:.2f}) fills/iter {r['fills_per_iter']:.0f} pwin {r['pwin']:.3f} | prior probe: {lz['per_trade_p50']} ({lz['p5']}, {lz['p95']}) | match {r['matches_prior_probe_to_cent']}")

def null2_run(mode, n, iters=ITERS, seed=SEED, side=None, label='', exclude_days=None):
    """exclude_days: DIAGNOSTIC only -- drop these entry days from the pool (used to apply his
    both-touched tie-break, the both-touched skip in williams_screen.backtest_fig41, as a sensitivity; the registered NULL-2 passes None)."""
    tab = TABLES[mode][1]
    kinds = TABLES[mode][2]
    pd_, pdir = ([], [])
    excl = np.zeros(N, dtype=bool)
    if exclude_days is not None:
        excl[np.asarray(exclude_days)] = True
    for d in (1, -1) if side is None else (side,):
        ok_d = ~np.isnan(tab[d][DAYS]) & ~excl[DAYS]
        pd_.append(DAYS[ok_d])
        pdir.append(np.full(int(ok_d.sum()), d))
    pday = np.concatenate(pd_)
    pdir = np.concatenate(pdir)
    P = len(pday)
    assert n <= len(np.unique(pday)), f'pool of days ({len(np.unique(pday))}) smaller than n={n}'
    rng = np.random.default_rng(seed)
    nets, pwins, nstop0, nstop, nbail = ([], [], [], [], [])
    for i in range(iters):
        perm = rng.permutation(P)
        dseq = pday[perm]
        _, first = np.unique(dseq, return_index=True)
        sel = perm[np.sort(first)[:n]]
        assert len(sel) == n
        v = np.where(pdir[sel] == 1, tab[1][pday[sel]], tab[-1][pday[sel]])
        k = np.where(pdir[sel] == 1, kinds[1][pday[sel]], kinds[-1][pday[sel]])
        nets.append(float(v.sum()))
        pwins.append(float((v > 0).mean()))
        nstop0.append(int((k == 'stop0').sum()))
        nstop.append(int((k == 'stop').sum()))
        nbail.append(int((k == 'bailout').sum()))
        if (i + 1) % 50 == 0:
            log(f'  NULL-2 {mode}{label}: {i + 1} of {iters} iterations | running net p50 {np.median(nets):,.2f}')
    nets = np.array(nets)
    q = pctl(nets)
    return ({'n_fills_per_iter': n, 'iters': iters, 'seed': seed, 'pool_pairs': int(P), 'pool_days': int(len(np.unique(pday))), 'net': q, 'per_trade': {k: q[k] / n for k in ('p5', 'p50', 'p95', 'mean', 'min', 'max')}, 'pwin_mean': float(np.mean(pwins)) * 100.0, 'kinds_per_iter_mean': {'entry_day_stop': float(np.mean(nstop0)), 'stop': float(np.mean(nstop)), 'bailout': float(np.mean(nbail))}}, nets)
NULL2 = {}
for mode in ('money', 'range'):
    n = REAL[mode]['n']
    log(f'NULL-2 {mode} stop: {ITERS} iterations at n={n} ...')
    r, nets = null2_run(mode, n)
    r['real_rule'] = {'pointer': REAL_PTR[mode], 'net': REAL[mode]['net'], 'per_trade': REAL[mode]['per_trade'], 'percentile_in_null2': rank_pct(nets, REAL[mode]['net'])}
    for name, d in (('long_only', 1), ('short_only', -1)):
        nn = REAL[mode][name.split('_')[0]]['n']
        rr, dn = null2_run(mode, nn, side=d, label=' ' + name)
        rr['real_rule_side'] = dict(REAL[mode][name.split('_')[0]])
        rr['real_rule_side']['percentile_in_null2_side'] = rank_pct(dn, REAL[mode][name.split('_')[0]]['net'])
        r[name] = rr
    NULL2[mode] = r
    log(f"NULL-2 {mode}: per-trade p5/p50/p95 {r['per_trade']['p5']:.2f} / {r['per_trade']['p50']:.2f} / {r['per_trade']['p95']:.2f} | net p50 {r['net']['p50']:,.0f} | real per {REAL[mode]['per_trade']:.2f} at pct {r['real_rule']['percentile_in_null2']['pct_below']:.1f} inside {r['real_rule']['percentile_in_null2']['inside_p5_p95']}")
BT_DAYS = DAYS[up_t & dn_t]
ST_UP = DAYS[up_t & ~dn_t]
ST_DN = DAYS[dn_t & ~up_t]
DIAG = {'both_touched_days': int(len(BT_DAYS)), 'single_touch_up_days': int(len(ST_UP)), 'single_touch_down_days': int(len(ST_DN))}
for mode in ('money', 'range'):
    tab = TABLES[mode][1]
    n = REAL[mode]['n']
    d = {'mean_pnl_per_fill': {'both_touched_long': float(np.nanmean(tab[1][BT_DAYS])), 'both_touched_short': float(np.nanmean(tab[-1][BT_DAYS])), 'single_touch_long': float(np.nanmean(tab[1][ST_UP])), 'single_touch_short': float(np.nanmean(tab[-1][ST_DN]))}}
    log(f"DIAG {mode}: mean P&L per fill both-touched long {d['mean_pnl_per_fill']['both_touched_long']:,.0f} short {d['mean_pnl_per_fill']['both_touched_short']:,.0f} | single-touch long {d['mean_pnl_per_fill']['single_touch_long']:,.0f} short {d['mean_pnl_per_fill']['single_touch_short']:,.0f}")
    r, nets = null2_run(mode, n, exclude_days=BT_DAYS, label=' [both-touched EXCLUDED]')
    r['real_rule'] = {'pointer': REAL_PTR[mode], 'net': REAL[mode]['net'], 'per_trade': REAL[mode]['per_trade'], 'percentile_in_this_variant': rank_pct(nets, REAL[mode]['net'])}
    for name, dd in (('long_only', 1), ('short_only', -1)):
        nn = REAL[mode][name.split('_')[0]]['n']
        rr, dn = null2_run(mode, nn, side=dd, label=' ' + name + ' [both-touched EXCLUDED]', exclude_days=BT_DAYS)
        rr['real_rule_side'] = dict(REAL[mode][name.split('_')[0]])
        rr['real_rule_side']['percentile_in_this_variant'] = rank_pct(dn, REAL[mode][name.split('_')[0]]['net'])
        r[name] = rr
    d['null2_both_touched_excluded'] = r
    DIAG[mode] = d
    log(f"DIAG {mode} NULL-2 [both-touched EXCLUDED]: per-trade p5/p50/p95 {r['per_trade']['p5']:.2f} / {r['per_trade']['p50']:.2f} / {r['per_trade']['p95']:.2f} | real {REAL[mode]['per_trade']:.2f} at pct {r['real_rule']['percentile_in_this_variant']['pct_below']:.1f} | long-only p50 {r['long_only']['per_trade']['p50']:.2f} (real long {REAL[mode]['long']['per_trade']:.2f} at p{r['long_only']['real_rule_side']['percentile_in_this_variant']['pct_below']:.1f}) | short-only p50 {r['short_only']['per_trade']['p50']:.2f} (real short {REAL[mode]['short']['per_trade']:.2f} at p{r['short_only']['real_rule_side']['percentile_in_this_variant']['pct_below']:.1f})")
OUT = {'generated': str(date.today()), 'label': 'POST-HOC same-fill control (NULL-2) -- A04 lab addendum 2026-09-18; NOT a pre-registered arm; the clause-B at-open control stays the registered one', 'seed': SEED, 'iters': ITERS, 'walk_forward_cap_bars': CAP, 'friction_rt': FRIC, 'money_stop_usd': ws.MONEY_STOP, 'pt_value': PT, 'data': {'file': 'data/ZB_F.parquet', 'rows': int(N), 'first': str(zb.index.min().date()), 'last': str(zb.index.max().date()), 'network': False}, 'engine': {'file': 'williams_screen.py', 'modified': False, 'imported': ['fetch', 'backtest_fig41', 'shuffle_fig41', 'SEED', 'SHUFFLE_ITERS', 'MONEY_STOP', 'SPEC']}, 'conventions': {'fill': 'entry at O[t] + d*(H[t-1]-L[t-1]); fill requires H[t] >= level (d=+1) or L[t] <= level (d=-1) -- williams_screen.backtest_fig41, the breakout-level and touch test (a day on which both levels are touched is skipped)', 'entry_day_stop': 'williams_screen.backtest_fig41, the entry-day stop branch (fires only on a clean reversal): same-day stop only on a clean reversal (stop on the far side of the open)', 'stop_modes': {'money': '$1,500 = 1.5 pts', 'range': '0.5 x previous-day range'}, 'exit': 'first favourable open (bailout) checked before the stop on each later bar -- williams_screen.backtest_fig41 and shuffle_fig41, both of which check the favourable open before the stop on each later bar', 'both_touched_day_skip_applied': False, 'one_position_at_a_time_applied': False, 'note_on_those_two': 'the as-run clause-B control applies neither either (williams_screen.shuffle_fig41); the real rule applies both (the both-touched skip and the single-open-position guard, both in williams_screen.backtest_fig41)', 'eligible_days': 't in [1, N-2] (williams_screen.shuffle_fig41, which samples entry days from range(1, N-1))', 'sampling_at_real_n': "uniform over the pool of touched (day, side) pairs, at most one fill per day, first n of a random permutation; per-fill distribution identical to the probe's (random day x fair coin x touch-required is uniform over the same pool); the probe's one-coin-per-day sampler cannot reach n (see pool.expected_fills_one_coin_per_day_all_days)", 'direction_caveat': "conditional on the fill, the coin's direction IS the breakout direction on every single-touch day (2,055 of 2,120 touchable days) and a fair coin only on the 65 both-touched days; NULL-2 therefore removes his SEQUENCING (one-position-at-a-time, both-touched skip) from his own fill population, not the breakout direction", 'per_trade_convention': 'net / n with n the fills per iteration (exactly n for NULL-2; the sampled entry count for the as-run control)'}, 'pool': POOL, 'unresolved_in_cap': {m: {'at_open_pairs': TABLES[m][3], 'same_fill_pairs': TABLES[m][4]} for m in ('money', 'range')}, 'probe_replication_60x3000': PROBE, 'real_rule': REAL, 'as_run_at_open_control': AS_RUN, 'null2': NULL2, 'diagnostic_both_touched_convention': DIAG, 'runtime_s': round(time.time() - T0, 1)}
path = os.path.join(HERE, 'results_null2.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(OUT, f, indent=2)
log(f'wrote {path}')
print()
print('=' * 118)
print(f"{'stop':6} | {'NULL-2 per-trade p5 / p50 / p95':34} | {'NULL-2 net p5 / p50 / p95':34} | {'REAL per (pct in NULL-2)':26} | at-open p50 per")
print('-' * 118)
for m in ('money', 'range'):
    r = NULL2[m]
    pt_ = r['per_trade']
    nt = r['net']
    rp = r['real_rule']['percentile_in_null2']
    print(f"{m:6} | {pt_['p5']:9.2f} / {pt_['p50']:9.2f} / {pt_['p95']:9.2f} | {nt['p5']:10,.0f} / {nt['p50']:10,.0f} / {nt['p95']:10,.0f} | {REAL[m]['per_trade']:8.2f} (p{rp['pct_below']:5.1f}, {('in' if rp['inside_p5_p95'] else 'OUT')}) | {AS_RUN[m]['per_trade']['p50']:8.2f}")
print('-' * 118)
for m in ('money', 'range'):
    for s in ('long_only', 'short_only'):
        r = NULL2[m][s]
        rs = r['real_rule_side']
        print(f"{m:6} {s:10} NULL-2 per p50 {r['per_trade']['p50']:8.2f} (p5 {r['per_trade']['p5']:8.2f}, p95 {r['per_trade']['p95']:8.2f}) n={r['n_fills_per_iter']:4d} | REAL side per {rs['per_trade']:8.2f} at p{rs['percentile_in_null2_side']['pct_below']:5.1f} | at-open {s} per p50 {AS_RUN[m][s]['per_trade_p50']:8.2f}")
print('=' * 118)
print(f"probe replication: money {PROBE['money']['matches_prior_probe_to_cent']} range {PROBE['range']['matches_prior_probe_to_cent']} | as-run json repro: money {AS_RUN['money']['reproduces_json_to_cent']} range {AS_RUN['range']['reproduces_json_to_cent']} | runtime {OUT['runtime_s']}s")
