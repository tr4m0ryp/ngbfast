# Benchmark scope, cuts and results

Run on a shared 8-core host (load average 7 to 21 during the runs) with `OMP_NUM_THREADS=1` and at
most 2 joblib workers. Measured wall time of the final runs: concrete 2,799 s, yacht 1,941 s,
credit-g 711 s, small-n variant 135 s, low-signal variant 828 s, one-SE variant about 400 s:
**6,814 s (1.9 h) in total**, inside the 2.5 h budget.

Reproduce from the repository root (data is cached in `../_data`, raw per-call latencies in
`../_data/ngbfast_raw`, both outside git):

```
python -m bench.concrete && python -m bench.yacht --trials 20 && python -m bench.credit_g --trials 20
python -m bench.variants.small_n credit-g 100
python -m bench.variants.low_signal 0 0.1
python -m bench.variants.one_se
python -m bench.summarize
```

## What was cut from the evidence plan, and why

| plan item | what ran | why |
|---|---|---|
| datasets: 8 UCI sets, California housing, adult, credit-g, optional daily index returns | concrete, yacht (UCI, uci_datasets test masks), credit-g (OpenML 31) | budget; fits cost 10 to 16 s each at this load. concrete and yacht are the small-n UCI sets the plan names; credit-g is the binary low-signal case. energy has a script (`bench/energy.py`) but was not run. kin8nm, naval, power, protein, wine-red, California, adult and the time-series case were not run |
| 5 seeds | 3 splits x 3 seeds (split s uses seed s) | budget |
| grid: leaf {1,20,100,400} x minibatch {1,0.5,0.2} x depth {1,3}, 500 rounds | full grid, 1 seed (split 0) | budget; the grid answers "where do constant learners appear", which is deterministic given the bound |
| random search: 50 trials, 5-fold CV | concrete 30 trials, yacht and credit-g 20 trials; 3-fold CV; rounds picked from the staged CV curve | budget. Search runs on split 0's training rows only |
| latency: 200 repetitions per batch size, 1 / 100 / full test rows | tuned models: 100 reps at 1 row, 50 at 100 rows, 30 at the full test set; default config: 100 reps at 1 row; grid: 15 reps at 1 row (arms pristine, fold, ours) | budget. yacht's test set is 30 rows, so it has no 100-row batch |
| external speed reference (LightGBMLSS / XGBoostLSS) | not run | not installed; it is a different model and was a reference only |
| `OMP_NUM_THREADS=1` pinned, load recorded | done; load recorded per run in each JSON (`load_start`, `load_end`) | |

Two failures were fixed during the run and are part of the code: upstream NGBoost with a Bernoulli
distribution can diverge (non-finite natural gradient) at large learning rates, which now marks the
trial as failed (no trial failed in the final runs); upstream also stops before `n_estimators` when
the projected gradient vanishes, so the CV curve is padded with its last value.

## Statistics

- **Equality:** every arm is compared bit for bit with upstream `NGBoost.pred_param` on every test
  row of every fitted model (grid, tuned and default; 30 models per dataset).
- **Latency:** arms run interleaved in a rotating order within each repetition. The paired
  statistic is `d = log t_ours - log t_arm` per repetition; within a model its median has a
  moving-block bootstrap SE (blocks of 10 repetitions, 2,000 resamples), because consecutive
  repetitions share machine load. Across the 3 seeds the per-seed medians are averaged with
  SE = sd / sqrt(3). Speedup = exp(-mean d).
- **Accuracy (one-SE variant):** per-example held-out NLL difference, averaged over seeds when the
  test rows are shared, SE = sd / sqrt(n) over independent test rows.

## Results

**Correctness holds everywhere:** fold, direct traversal and both together are bitwise identical to
upstream on 100 % of 6,994 test rows over 90 fitted models (max |d theta| = 0.0). The merged-step
ablation is not: 73 to 80 % of rows bitwise, max |d theta| 4.0e-15.

**Tuned public configurations have no constant learners.** The constant share at the searched
configuration was 0 % on all 3 seeds of concrete, yacht and credit-g, and 0 % at ngboost's default
configuration. On the grid the share is never partial: it is exactly 0 or exactly 1, and it is 1
precisely where the split bound fails (each round sees fewer rows than 2 x min_samples_leaf): 6 of 24
grid points on concrete, 10 on yacht, 6 on credit-g. Those degenerate fits cost NLL (concrete 4.23
vs 2.88 for the best grid point; yacht 2.07 vs -0.80), which is why the search never picks them.

**So the tuned-model gain is direct traversal, not folding.** One-row prediction, ours vs arm,
mean over 3 seeds of the median paired log ratio (SE):

| dataset | vs upstream | vs fold only | vs direct only | vs merged ablation |
|---|---|---|---|---|
| concrete | 3.67x (log -1.300 +- 0.005) | 3.83x (-1.344 +- 0.005) | 1.01x (-0.007 +- 0.005) | 0.97x (+0.035 +- 0.004) |
| yacht | 3.60x (-1.280 +- 0.017) | 3.78x (-1.330 +- 0.009) | 1.01x (-0.005 +- 0.004) | 0.96x (+0.039 +- 0.002) |
| credit-g | 3.51x (-1.254 +- 0.004) | 3.52x (-1.259 +- 0.010) | 1.00x (+0.000 +- 0.002) | 0.95x (+0.052 +- 0.003) |

At 100 rows and at the full test set the ratio vs upstream is the same within 2 SE (concrete 3.94x
and 3.81x, yacht 3.86x at 30 rows, credit-g 3.57x at 100 rows). **Ours vs direct only is a tie** on
every dataset (|log ratio| <= 1.5 SE): with no constant learner, folding has nothing to remove.
Fold only is 4 % slower than upstream at share 0 (its per-round bookkeeping, not removed work).
The merged ablation is 3 to 5 % faster than ours at share 0, where it does exactly the same learner
work: that gap is per-round overhead in ngbfast's prediction loop, not the price of exactness.

**Fold build time is not negligible:** building the plan took a median of 17 % (concrete), 19 %
(yacht) and 33 % (credit-g) of the fit time on these small data sets, against the plan's expected
< 1 %. It probes each learner on a zero row plus a NaN and an infinity row (three `predict` calls), so
it costs about three one-row upstream predictions; it pays for itself after about four one-row calls.

## Variants: what would have to be true for folding to matter (spec section 6)

The tuned configurations showed no constant learners, so three settings were tested in which the
best model could be (close to) a fixed distribution.

1. **Small n (credit-g subsampled to 100 training rows, 20-trial search).** The search picked
   max_depth 4, leaf 10, 3 rounds: split trees, constant share 0 on 3 of 3 seeds; ours vs upstream
   one-row about 2.5x. Low signal is absorbed by stopping after 3 rounds, not by constant learners.
   Constant-learner configurations were close (leaf 200 / 400: CV NLL 0.5511 vs best 0.5465).
   **Result: no constant learners at the tuned config.**
2. **Low signal (Friedman #1 with the signal scaled to a = 0 and a = 0.1 of the unit noise, 400
   training rows, 15-trial search).** The search picked leaf 200 with minibatch 0.5 (200 rows per
   round, 400 needed to split): every learner constant on 3 of 3 seeds at both levels. One-row
   prediction: upstream 320 to 394 ms p50 vs ours 0.18 ms; ours vs upstream 1,808x to 2,215x per seed
   (block-bootstrap SE of the log ratio 0.02 to 0.07, 60 repetitions each). Fold alone was faster
   still (0.16 ms), as the float32 input check of direct traversal is pure overhead here.
   **Result: when there is no usable signal, tuning lands on a fully constant model, and folding
   wins by three orders of magnitude, bit-identically.**
3. **One-SE selection (Breiman et al. 1984: the most regularised trial within one SE of the best CV
   NLL).** credit-g n = 100: the rule picks leaf 400 (all learners constant). Held out, its NLL is
   +0.020 +- 0.014 nats above the best pick (n = 100 test rows, 3 seeds averaged): a tie within 1.4
   SE, at 0.12 ms vs 0.47 ms one-row (ours on both; the best pick has only 3 rounds). Friedman a = 0
   and a = 0.1: the rule picks leaf 400, also all constant, tie with the best pick (-0.00012 +- 0.00016
   and -0.00011 +- 0.00012 nats, n = 200). **Result: a tie on accuracy; kept as a documented option,
   not adopted as a win, because the accuracy difference is not resolved at this n.**

## Verdict

- **Correctness (a): confirmed** on every model and row.
- **Gain as a function of constant share (b): degenerate in practice.** The share is 0 or 1. At 1
  the one-row gain (ours vs upstream, 15 repetitions, grid) is 4,742x to 10,069x on concrete and
  yacht, 964x to 1,842x on credit-g, 1,808x to 2,215x in the low-signal variant, and 7x to 8x on two
  yacht fits that upstream stopped after 1 round (the gain scales with the rounds removed). At 0
  folding gives 1.00x to 1.01x (a tie with direct traversal). Tuned models on the three public sets sit at 0; only no-signal or very small data puts
  the search at 1.
- **Direct traversal (c): the dependable gain**, 3.5x to 3.7x one-row on every tuned model, SE of the
  log ratio <= 0.017.
- As the spec's fallback says, the publishable part is the upstream patch plus the degeneracy
  diagnostic (it flags exactly the grid points where the bound fails), not a folding speedup on
  tuned models.
