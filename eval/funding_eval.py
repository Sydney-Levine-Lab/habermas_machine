# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluates the funding slate prompts on annotated scenarios.

Each run asks the slate model for NUM_SLATES slates once, without retries, and
scores the response on:

* Constraints: the model returned exactly NUM_SLATES slates, each slate passes
  funding.parse_slate, and no slate leaves money unspent when it could raise a
  funded proposal toward its ideal or fund another proposal.
* Consensus: each slate meets the scenario's hand-labeled consensus items.
* Diversity: the mean pairwise L1 distance between allocations (as a fraction
  of the largest possible distance), counting only good slates, i.e. slates
  that pass the constraints and the consensus. Missing or bad slates count as
  zero distance, so diversity can't be bought by breaking constraints.
* Judge: an LLM scores how well the slates cover the disagreements, fit the
  consensus, differ from each other, and stay faithful to the opinions.

Usage:
  set -a; . ./.env; set +a
  python eval/funding_eval.py --label v0 --out runs/funding_eval
  python eval/funding_eval.py --variant path/to/funding_slate_model.py ...
"""

import argparse
import collections
from concurrent import futures
import importlib.util
import itertools
import json
import os
import pathlib
import re
import statistics
import threading
import time

from habermas_machine import funding
from habermas_machine.llm_client import openai_client
from habermas_machine.statement_model import funding_slate_model

NUM_SLATES = 8
MODEL = 'gpt-6.1-sol'
GENERATION_EFFORT = 'medium'
JUDGE_EFFORT = 'high'
# USD per million tokens (input, output).
PRICES = (2.0, 10.0)
SCENARIOS = pathlib.Path(__file__).with_name('funding_scenarios.json')

_usage = threading.local()


def _record_usage(client: openai_client.OpenAIClient) -> None:
  """Wraps the client's API call so each thread can read its token usage."""
  create = client._client.chat.completions.create  # pylint: disable=protected-access

  def wrapped(*args, **kwargs):
    response = create(*args, **kwargs)
    _usage.value = response.usage
    return response

  client._client.chat.completions.create = wrapped  # pylint: disable=protected-access


def _sample(client, prompt, attempts=4, **kwargs):
  """Samples text, retrying transient API errors. Returns (text, usage)."""
  for attempt in range(attempts):
    try:
      _usage.value = None
      text = client.sample_text(prompt, **kwargs)
      usage = _usage.value
      details = getattr(usage, 'completion_tokens_details', None)
      return text, {
          'input': usage.prompt_tokens,
          'output': usage.completion_tokens,
          'reasoning': getattr(details, 'reasoning_tokens', 0) or 0,
      }
    except Exception as exc:  # pylint: disable=broad-except
      if attempt == attempts - 1:
        raise
      print(f'API error ({exc!r}), retrying.')
      time.sleep(5 * 2**attempt)


def load_module(path: str | None):
  """Loads a slate model module from a file, or the installed one."""
  if path is None:
    return funding_slate_model
  spec = importlib.util.spec_from_file_location(
      f'variant_{abs(hash(path))}', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _proposals(scenario):
  return [funding.Proposal(**p) for p in scenario['proposals']]


def _mode_data(scenario, mode):
  """Returns (previous_winner, critiques, consensus, disagreements)."""
  if mode == 'opinion_only':
    return None, None, scenario['consensus'], scenario['disagreements']
  round_ = scenario['critique_round']
  proposals = _proposals(scenario)
  previous = funding.parse_slate(
      json.dumps(round_['previous_slate']), proposals,
      scenario['funding_pool'])
  return (
      funding.render_slate(previous, proposals, scenario['funding_pool']),
      round_['critiques'],
      scenario['consensus'] + round_['consensus'],
      round_['disagreements'],
  )


def _reserve(consensus) -> int:
  return sum(c['amount'] for c in consensus if c['rule'] == 'reserve')


def consensus_failures(slate, consensus, funding_pool):
  """Returns the consensus items that the slate breaks."""
  amounts = {a.proposal_id: a.amount for a in slate.allocations}
  failures = []
  for item in consensus:
    rule = item['rule']
    if rule == 'funded':
      ok = amounts[item['id']] > 0
    elif rule == 'zero':
      ok = amounts[item['id']] == 0
    elif rule == 'at_least':
      ok = amounts[item['id']] >= item['amount']
    elif rule == 'reserve':
      ok = slate.total <= funding_pool - item['amount']
    else:
      raise ValueError(f'Unknown rule: {rule}')
    if not ok:
      failures.append(item)
  return failures


def unspent_violation(slate, proposals, consensus, funding_pool) -> bool:
  """True if money is left over that could raise or fund a proposal."""
  leftover = funding_pool - _reserve(consensus) - slate.total
  if leftover <= 0:
    return False
  vetoed = {c['id'] for c in consensus if c['rule'] == 'zero'}
  amounts = {a.proposal_id: a.amount for a in slate.allocations}
  for proposal in proposals:
    amount = amounts[proposal.id]
    if 0 < amount < proposal.ideal:
      return True
    if (amount == 0 and proposal.id not in vetoed
        and proposal.minimum <= leftover):
      return True
  return False


def _distance(a, b, funding_pool) -> float:
  """L1 distance between two slates over the largest possible, 2 * pool."""
  return sum(
      abs(x.amount - y.amount) for x, y in zip(a.allocations, b.allocations)
  ) / (2 * funding_pool)


def score_slates(slate_texts, scenario, consensus):
  """Scores parsed slates against the constraints, consensus, and diversity."""
  proposals = _proposals(scenario)
  pool = scenario['funding_pool']
  records = []
  for text in slate_texts[:NUM_SLATES]:
    record = {'text': text}
    try:
      slate = funding.parse_slate(text, proposals, pool)
    except ValueError as exc:
      record['error'] = str(exc)
      records.append(record)
      continue
    record['slate'] = slate
    record['amounts'] = [a.amount for a in slate.allocations]
    record['unspent'] = unspent_violation(slate, proposals, consensus, pool)
    record['consensus_failures'] = consensus_failures(slate, consensus, pool)
    records.append(record)

  valid = [r for r in records if 'slate' in r]
  constraint_ok = [r for r in valid if not r['unspent']]
  good = [r for r in constraint_ok if not r['consensus_failures']]
  pairs = list(itertools.combinations(good, 2))
  distances = [_distance(a['slate'], b['slate'], pool) for a, b in pairs]
  max_pairs = NUM_SLATES * (NUM_SLATES - 1) / 2
  return records, {
      'num_returned': len(slate_texts),
      'count_ok': len(slate_texts) == NUM_SLATES,
      'valid_rate': len(valid) / NUM_SLATES,
      'unspent_rate': (
          sum(r['unspent'] for r in valid) / len(valid) if valid else 0.0),
      'constraint_score': len(constraint_ok) / NUM_SLATES,
      'consensus_rate': (
          sum(not r['consensus_failures'] for r in valid) / len(valid)
          if valid else 0.0),
      'good_rate': len(good) / NUM_SLATES,
      # Missing and bad slates count as zero distance.
      'diversity': sum(distances) / max_pairs,
      'diversity_among_good': (
          statistics.mean(distances) if distances else 0.0),
      'min_distance': min(distances) if distances else 0.0,
      'distinct_funded_sets': len({
          tuple(amount > 0 for amount in r['amounts']) for r in good}),
  }


_JUDGE_PROMPT = """You are evaluating a set of alternative funding slates that an AI mediator wrote for a group at Edge City. The group will vote on the slates, so a good set gives the group real alternatives: every slate respects what everyone agrees on, and across the set the slates take the different sides of each real disagreement, roughly in proportion to how many people hold each side. Slates that differ only by small amounts are near-duplicates and are not real alternatives.

{fund}

Proposals:
{proposals}

Individual opinions:
{opinions}
{critique_section}
Reference notes from the evaluation designer (the slate writer did not see these):
Points everyone agrees on: {consensus}
Main disagreements:
{disagreements}

Slates to evaluate:

{slates}

Score the set of slates as a whole from 1 (very poor) to 10 (excellent) on each criterion:
- disagreement_coverage: Across the set, are the main sides of each disagreement represented, in rough proportion to their support? Is any substantial view left out entirely?
- consensus_fit: Do all slates respect what everyone agrees on, including softer agreements that the reference notes do not list?
- distinctness: Are the slates meaningfully different alternatives that a participant would actually choose between, rather than near-duplicates or arbitrary variations no one asked for?
- faithfulness: Do the summaries and reasons accurately reflect what people said, without claiming support no one expressed, and do they match the amounts?

Respond with brief reasoning, then a JSON object inside <json></json> tags with integer scores and a one-sentence note on the biggest weakness:
<json>{{"disagreement_coverage": 0, "consensus_fit": 0, "distinctness": 0, "faithfulness": 0, "weakness": "..."}}</json>"""


def _describe_consensus(consensus, proposals):
  titles = {p.id: p.title for p in proposals}
  parts = []
  for c in consensus:
    if c['rule'] == 'funded':
      parts.append(f'fund {c["id"]} ({titles[c["id"]]})')
    elif c['rule'] == 'zero':
      parts.append(f'give {c["id"]} ({titles[c["id"]]}) $0')
    elif c['rule'] == 'at_least':
      parts.append(
          f'give {c["id"]} ({titles[c["id"]]}) at least'
          f' {funding.format_amount(c["amount"])}')
    elif c['rule'] == 'reserve':
      parts.append(
          f'keep {funding.format_amount(c["amount"])} unallocated in reserve')
  return '; '.join(parts) or 'nothing specific'


def judge(client, scenario, mode, records):
  """Asks the judge to score the valid slates as a set."""
  proposals = _proposals(scenario)
  pool = scenario['funding_pool']
  previous, critiques, consensus, disagreements = _mode_data(scenario, mode)
  valid = [r for r in records if 'slate' in r]
  if not valid:
    return None, None
  slates = '\n\n'.join(
      f'Slate {i + 1}:\n{funding.render_slate(r["slate"], proposals, pool)}'
      for i, r in enumerate(valid))
  critique_section = ''
  if previous is not None:
    critique_section = (
        f'\nSlate from the previous round:\n{previous}\n\nCritiques of that'
        ' slate:\n' + '\n'.join(
            f'Person {i + 1}: {c}' for i, c in enumerate(critiques)) + '\n')
  prompt = _JUDGE_PROMPT.format(
      fund=funding.fund_description(pool),
      proposals=funding.format_proposals(proposals),
      opinions='\n'.join(
          f'Person {i + 1}: {o}' for i, o in enumerate(scenario['opinions'])),
      critique_section=critique_section,
      consensus=_describe_consensus(consensus, proposals),
      disagreements='\n'.join(f'- {d}' for d in disagreements),
      slates=slates,
  )
  text, usage = _sample(
      client, prompt, max_tokens=32000, timeout=600, temperature=0.0)
  match = re.search(r'<json>\s*(\{.*?\})\s*</json>', text, re.DOTALL)
  if not match:
    return {'error': 'no json', 'raw': text[-2000:]}, usage
  scores = json.loads(match.group(1))
  keys = ['disagreement_coverage', 'consensus_fit', 'distinctness',
          'faithfulness']
  scores['mean'] = statistics.mean(scores[k] for k in keys)
  return scores, usage


def run_one(module, gen_client, judge_client, scenario, mode, sample):
  """Generates and scores one set of slates."""
  proposals = _proposals(scenario)
  pool = scenario['funding_pool']
  previous, critiques, consensus, _ = _mode_data(scenario, mode)
  prompt = module._generate_prompt(  # pylint: disable=protected-access
      funding.default_question(pool), scenario['opinions'], NUM_SLATES,
      proposals, pool, previous, critiques)
  start = time.time()
  response, gen_usage = _sample(
      gen_client, prompt, max_tokens=module._MAX_TOKENS,  # pylint: disable=protected-access
      terminators=['</answer>'], timeout=module._TIMEOUT_SECONDS)  # pylint: disable=protected-access
  seconds = time.time() - start
  slate_texts, explanation = module._process_model_response(response)  # pylint: disable=protected-access
  records, metrics = score_slates(slate_texts, scenario, consensus)
  judge_scores, judge_usage = (
      judge(judge_client, scenario, mode, records) if judge_client
      else (None, None))
  return {
      'scenario': scenario['id'],
      'mode': mode,
      'sample': sample,
      'metrics': metrics,
      'judge': judge_scores,
      'usage': {'generation': gen_usage, 'judge': judge_usage},
      'seconds': seconds,
      'explanation': explanation,
      'response': response,
      'slates': [
          {k: v for k, v in r.items() if k != 'slate'} for r in records],
  }


def cost(usage) -> float:
  if not usage:
    return 0.0
  return (usage['input'] * PRICES[0] + usage['output'] * PRICES[1]) / 1e6


def summarize(results):
  """Returns summary lines with means overall, per mode, and per scenario."""
  metric_keys = ['count_ok', 'valid_rate', 'unspent_rate',
                 'constraint_score', 'consensus_rate', 'good_rate',
                 'diversity', 'diversity_among_good', 'min_distance',
                 'distinct_funded_sets']
  judge_keys = ['disagreement_coverage', 'consensus_fit', 'distinctness',
                'faithfulness', 'mean']

  def means(group):
    row = {k: statistics.mean(float(r['metrics'][k]) for r in group)
           for k in metric_keys}
    judged = [r['judge'] for r in group if r['judge'] and 'mean' in r['judge']]
    for k in judge_keys:
      row[f'judge_{k}'] = (
          statistics.mean(j[k] for j in judged) if judged else float('nan'))
    return row

  groups = collections.defaultdict(list)
  for r in results:
    groups['ALL'].append(r)
    groups[r['mode']].append(r)
    groups[r['scenario']].append(r)
  lines = []
  header = ['group', 'n', 'cnt', 'valid', 'unsp', 'constr', 'cons', 'good',
            'div', 'divG', 'minD', 'sets', 'jCov', 'jCons', 'jDist', 'jFaith',
            'jMean']
  lines.append(' '.join(f'{h:>8}' for h in header))
  for name, group in groups.items():
    row = means(group)
    values = [row[k] for k in metric_keys] + [
        row[f'judge_{k}'] for k in judge_keys]
    lines.append(f'{name[:8]:>8} {len(group):>8} ' + ' '.join(
        f'{v:>8.3f}' for v in values))
  total = sum(cost(r['usage']['generation']) + cost(r['usage']['judge'])
              for r in results)
  gen_out = statistics.mean(r['usage']['generation']['output'] for r in results)
  lines.append(
      f'cost ${total:.2f}; mean generation output tokens {gen_out:.0f};'
      f' mean seconds {statistics.mean(r["seconds"] for r in results):.0f}')
  return lines


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--variant', help='Path to a slate model module.')
  parser.add_argument('--label', required=True)
  parser.add_argument('--out', default='runs/funding_eval')
  parser.add_argument('--samples', type=int, default=3)
  parser.add_argument('--scenarios', nargs='*')
  parser.add_argument('--modes', nargs='*',
                      default=['opinion_only', 'critique'])
  parser.add_argument('--no-judge', action='store_true')
  parser.add_argument('--workers', type=int, default=24)
  args = parser.parse_args()

  module = load_module(args.variant)
  scenarios = json.loads(SCENARIOS.read_text())
  if args.scenarios:
    scenarios = [s for s in scenarios if s['id'] in args.scenarios]
  gen_client = openai_client.OpenAIClient(
      MODEL, reasoning_effort=GENERATION_EFFORT)
  _record_usage(gen_client)
  judge_client = None
  if not args.no_judge:
    judge_client = openai_client.OpenAIClient(
        MODEL, reasoning_effort=JUDGE_EFFORT)
    _record_usage(judge_client)

  jobs = [(s, m, i) for s in scenarios for m in args.modes
          for i in range(args.samples)]
  results = []
  with futures.ThreadPoolExecutor(args.workers) as pool:
    pending = {
        pool.submit(run_one, module, gen_client, judge_client, s, m, i):
        (s['id'], m, i) for s, m, i in jobs}
    for future in futures.as_completed(pending):
      try:
        results.append(future.result())
      except Exception as exc:  # pylint: disable=broad-except
        print(f'Run {pending[future]} failed: {exc!r}')
      print(f'{len(results)}/{len(jobs)} done', flush=True)

  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  results.sort(key=lambda r: (r['scenario'], r['mode'], r['sample']))
  with open(out / f'{args.label}.jsonl', 'w') as f:
    for r in results:
      f.write(json.dumps(r) + '\n')
  lines = summarize(results)
  (out / f'{args.label}.summary.txt').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))


if __name__ == '__main__':
  main()
