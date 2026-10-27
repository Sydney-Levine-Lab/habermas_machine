# Funding slate prompt evaluation

`funding_eval.py` scores the `FundingSlateModel` prompts on the hand-annotated
scenarios in `funding_scenarios.json`. Each of the 8 scenarios has 6–8
proposals, labeled consensus items (`funded`, `zero`, `at_least`, `reserve`),
the main disagreements, and a critique round with a previous slate.

```sh
set -a; . ./.env; set +a
PYTHONPATH=. python eval/funding_eval.py --label my_run          # current prompts
PYTHONPATH=. python eval/funding_eval.py --variant path/to/funding_slate_model.py --label alt
```

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

Faithfulness trades off against diversity. Every version that made the slates
more diverse lost some judged faithfulness, mostly because some slates give
more than the ideal to proposals nobody asked to expand. A tighter ideal cap
restored faithfulness, but the model then repeated slates or returned fewer
than 8.
