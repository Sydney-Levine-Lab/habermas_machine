# Funding slate prompt evaluation

`funding_eval.py` scores the `FundingSlateModel` prompts on the hand-annotated
scenarios in `funding_scenarios.json`. Each of the 8 scenarios has 6–8
proposals, labeled consensus items (`funded`, `zero`, `at_least`, `reserve`),
the main disagreements, and a critique round with a previous slate.

`funding_scenarios_heldout.json` has 4 more scenarios that were only used to
check the final prompts, not to tune them.

```sh
set -a; . ./.env; set +a
PYTHONPATH=. python eval/funding_eval.py --label my_run          # current prompts
PYTHONPATH=. python eval/funding_eval.py --variant path/to/funding_slate_model.py --label alt
PYTHONPATH=. python eval/funding_eval.py --scenario-file eval/funding_scenarios_heldout.json --label heldout
PYTHONPATH=. python eval/funding_eval.py --oversample 1.5 --label oversampled
```

A critique round marked `padding_allowed` has fewer than 8 distinct slates that
spend the whole fund, so slates that leave money unspent don't count against
it.

Each run makes one call for 8 slates with `gpt-6.1-sol` at medium reasoning
effort, without retries, and scores it on:

| Column | Meaning |
|---|---|
| `cnt` | Returned exactly 8 slates. |
| `valid` | Slates that pass `funding.parse_slate`, out of 8. |
| `unsp` | Valid slates that leave money unspent when a funded proposal is below its ideal or another non-vetoed proposal would fit. |
| `constr` | Slates that are valid and not unspent, out of 8. |
| `cons` | Valid slates that meet every labeled consensus item. |
| `div` | Mean pairwise L1 distance over 2 × pool between good slates, with missing or bad slates counting as zero. |
| `jCov`, `jCons`, `jDist`, `jFaith` | Scores from 1 to 10 by a `gpt-6.1-sol` judge at high effort: disagreement coverage, consensus fit, distinctness, and faithfulness. |

## Results

There were 48 runs per version: 8 scenarios × 2 modes × 3 samples. The full
iteration log is in the commit message.

| Version | constr | div | jCov | jCons | jDist | jFaith | jMean |
|---|---|---|---|---|---|---|---|
| Original prompts | 0.911 | 0.225 | 6.40 | 9.79 | 5.79 | 9.28 | 7.81 |
| Current prompts | 0.995 | 0.337 | 7.23 | 9.54 | 6.25 | 7.90 | 7.73 |

The original prompts returned fewer than 8 slates in 12.5% of calls, because
"stay at or below the ideal" and "use the whole fund" left too few
allocations. Their slates were often near-duplicates or left out one side of a
disagreement. The current prompts add guidance on how slates should differ:
fix the agreed points, give every side a clear slate, and keep slates at least
10% of the pool apart. They also require exactly 8 slates and let slates go
above an ideal when people asked for more.

A second round targeted constraints of at least 0.99 and faithfulness of at
least 7.9, then as much diversity and distinctness as possible:

| Version | Scenarios | constr | div | jCov | jDist | jFaith | jMean |
|---|---|---|---|---|---|---|---|
| Round 1 prompts | held-out | 0.927 | 0.278 | 7.04 | 6.17 | 9.00 | 8.00 |
| Current | held-out | 1.000 | 0.274 | 7.00 | 5.92 | 9.33 | 8.02 |
| Round 1 prompts | 3 tightest dev | 1.000 | 0.413 | 7.72 | 5.22 | 7.33 | 7.56 |
| Current | 3 tightest dev | 1.000 | 0.358 | 6.61 | 4.61 | 8.28 | 7.33 |

The current version meets both floors in both sets; round 1 missed the
constraint floor on held-out and the faithfulness floor on the tightest dev
scenarios. Diversity didn't improve: the current version is about even on
held-out and lower on the tight scenarios. Changes in round 2:

- Reasons must say when an amount is above the ideal and who asked for it, and
  describe support accurately ("one member", not "the group").
- The 10% difference between slates is a goal, not a rule, so the model writes
  closer slates instead of fewer.
- If the model still returns too few distinct slates, code pads the rest by
  giving one contested proposal its minimum and leaving the money unallocated.

Tried and dropped (on the dev set):

- Writing 12 slates and picking the 8 most spread out raised distance (+0.04)
  but lowered judged coverage in 12 of 16 cells, because it drops compromise
  and majority-side slates. Proportional quotas per side didn't fix that. It
  remains available as `oversample`, off by default.
- Planning the slates in a separate call first matched one-call selection on
  diversity, with lower coverage and about 1.4× the time and cost.
- Telling the model it may leave money unallocated when needed made it do so
  where 8 fully spent slates were possible (constraints fell to 0.90), even
  with stricter wording. That's why padding happens in code.

Faithfulness trades off against diversity. Every version that made the slates
more diverse lost some judged faithfulness, mostly because some slates give
more than the ideal to proposals nobody asked to expand. A tighter ideal cap
restored faithfulness, but the model then repeated slates or returned fewer
than 8.
