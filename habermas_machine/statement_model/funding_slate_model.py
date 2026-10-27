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

"""A chain-of-thought model that generates diverse funding slates."""

from collections.abc import Sequence
import dataclasses
import json
import math
import re

from habermas_machine import funding
from habermas_machine.llm_client import base_client
from habermas_machine.statement_model import base_model
from habermas_machine.statement_model import utils

# Generating many slates in one call needs a long response.
_MAX_TOKENS = 16384
_TIMEOUT_SECONDS = 300

# The example uses its own small fund so the model doesn't copy its amounts.
_EXAMPLE_PROPOSALS = """Proposal P1: Sunrise yoga on the beach
Minimum: $200
Ideal: $400
Description: Buy mats and pay a local teacher for daily sunrise yoga.

Proposal P2: Weekend trip to Hampi
Minimum: $600
Ideal: $900
Description: A bus and guide for a weekend trip to the ruins at Hampi.

Proposal P3: Shared tool library
Minimum: $150
Ideal: $300
Description: A shelf at The Circle with tools, adapters, and cables that anyone can borrow."""

_EXAMPLE_SLATE_1 = """{"side": "Trip first", "summary": "Fund the Hampi trip that most people asked for, plus the tool library, which everyone supports.", "allocations": [{"id": "P1", "amount": 0, "reason": "Only one person asked for yoga, and the trip needs most of the fund."}, {"id": "P2", "amount": 700, "reason": "Most people want the trip; this covers the bus and part of the guide."}, {"id": "P3", "amount": 300, "reason": "Everyone supports the tool library, so it gets its ideal amount."}]}"""

_EXAMPLE_SLATE_2 = """{"side": "Trip and yoga", "summary": "Fund more projects at smaller amounts so the money reaches more people.", "allocations": [{"id": "P1", "amount": 250, "reason": "Daily yoga is open to everyone and is cheap to run with fewer mats."}, {"id": "P2", "amount": 600, "reason": "The trip gets its minimum, which covers the bus without a guide."}, {"id": "P3", "amount": 150, "reason": "Everyone supports the tool library; this covers a basic shelf."}]}"""

_RULES = """Every slate must follow these rules:
- Include every proposal exactly once, using only the proposal IDs listed below. Do not add, rename, merge, or change proposals.
- Give each proposal either $0 or at least its minimum amount. Never give a proposal more than $0 but less than its minimum.
- Try to stay at or below each proposal's ideal amount. Give more than the ideal only when someone asked for more of that proposal or it serves what people said (for example, more spots). The reason must then say that the amount is above the ideal, what the extra money adds, and who asked for it.
- Describe support accurately in summaries and reasons. Say how many people hold a view (for example, "one member" or "most members"), and never present one person's view as the whole group's.
- Use whole dollars.
- The total must not be more than {pool}. Use the whole fund: if money is left over, raise funded proposals toward their ideal amounts or fund another proposal. Leave money unallocated only for a specific reason, such as the group asking to keep some money back, or every funded proposal being at its ideal and no remaining proposal that the group has not rejected fitting in what is left, and say what that reason is in the summary.
- Give a one-sentence reason for every amount, including $0."""

_DIVERSITY = """How to make the {num_statements} slates differ:
- Points that everyone agrees on are fixed. Every slate must follow them, and no slate should cut a proposal that everyone supports below what people asked for just to create variety.
- Find each real disagreement, its sides, and how many people hold each side.
- For every side of a disagreement, include at least one slate that gives that side what it asked for as fully as the fund and the agreed points allow, even if only one or two people hold it. Do not let compromises crowd out these clear options.
- Use the remaining slates for compromises and other mixes, giving more slates to views with more support.
- Make every slate a distinct alternative that someone in the group would prefer to all the others. Two slates that fund the same proposals should move at least {min_shift} between proposals; small adjustments are not new alternatives. This is a goal, not a rule: if the rules leave too few options, write slates with smaller differences rather than fewer slates.
- If members mostly agree, differ on other choices that matter, such as whether to fully fund a few projects or fund more projects at smaller amounts, and which borderline proposals to include.
- Always write exactly {num_statements} slates. If the rules leave few allocations open, create alternatives by choosing which supported proposals to fund, funding some at their minimum and others at their ideal, or giving more than the ideal to proposals where people asked for more (for example, more spots or more sessions).{extra}"""

_CRITIQUE_EXTRA = """
- If some members were content with the previous draft, include one slate that stays close to it while making the changes that everyone asked for."""


def _min_shift(funding_pool: int) -> str:
  """Smallest change that makes two slates real alternatives."""
  return funding.format_amount(max(50, round(funding_pool * 0.1 / 50) * 50))


_SIDE = """The "side" is a few words naming the view or direction that the slate represents, such as "Trip first" or "Compromise: trip and yoga". Slates that represent the same view must use exactly the same side."""

_FORMAT = """Write each slate as a JSON object with this shape:
{{"side": "<a few words naming the view this slate represents>", "summary": "<one or two sentences on the overall approach>", "allocations": [{{"id": "<proposal ID>", "amount": <whole dollars>, "reason": "<one sentence>"}}, ...]}}
""" + _SIDE + """

Provide your answer in the following format, with exactly {num_statements} slates separated by <sep>:
<answer>
[Your concise step-by-step reasoning, including a one-line plan for each slate]
<sep>
[Funding slate 1 as JSON]
<sep>
[Funding slate 2 as JSON]
<sep>
...
<sep>
[Funding slate {num_statements} as JSON]
</answer>"""


def _example(reasoning: str) -> str:
  """Formats the worked example with the given reasoning."""
  return f"""Example (with a $1,000 fund, three proposals, and 2 slates):

Proposals:
{_EXAMPLE_PROPOSALS}

<answer>
{reasoning}
<sep>
{_EXAMPLE_SLATE_1}
<sep>
{_EXAMPLE_SLATE_2}
</answer>"""



def _generate_opinion_critique_prompt(
    question: str,
    opinions: Sequence[str],
    previous_winner: str,
    critiques: Sequence[str],
    num_statements: int,
    proposals: Sequence[funding.Proposal],
    funding_pool: int,
    context: str | Sequence[str | None] | None = None,
) -> str:
  """Generates a prompt using opinions, previous winner, and critiques."""
  context_section = utils.format_context(context)
  context_description = (
      ' context about the participants,' if context_section else '')
  pool = funding.format_amount(funding_pool)
  example = _example(
      """1. Everyone supports the tool library (Opinions 1, 2, 3). Most people want the Hampi trip (Opinions 1, 3), and Opinion 2 wants yoga instead.
2. The previous draft funded the trip at its ideal and gave nothing to the other proposals.
3. Critique 2 says the tool library should be funded since everyone wants it. Critique 3 says the trip doesn't need a guide. Critique 1 accepts the trip but would like some yoga.
4. The revised slates should all fund the tool library and the trip. They should differ on whether to fund yoga by giving the trip less.
5. Slate 1 keeps the trip above its minimum and gives the tool library its ideal (Critique 2). Slate 2 gives the trip its minimum, without a guide (Critique 3), so that yoga can be funded (Opinion 2, Critique 1). Both total $1,000.""")

  prompt = f"""You are helping a group of people at Edge City decide how to split a community fund. {funding.SETTING}

{funding.fund_description(funding_pool)}

Each person in the group shared their opinion, someone wrote a first draft of a funding slate, and group members then critiqued that draft. A funding slate says how much money each proposal gets and why. Your job is to generate {num_statements} revised funding slates that use this feedback to better reflect what the group wants. The group will vote on these slates, so make them real alternatives that differ clearly from each other.

Base the slates on what the group members said. Use proposal details, such as cost and how realistic a project is, only to settle choices that the opinions and critiques leave open, and never claim support that no one expressed.

{_RULES.format(pool=pool)}

{_DIVERSITY.format(num_statements=num_statements, min_shift=_min_shift(funding_pool), extra=_CRITIQUE_EXTRA)}

Please think through this task step-by-step:

1. Read the individual opinions carefully. Note which proposals people support or oppose, where they agree, and where they disagree.
2. Review the previous draft funding slate. Note what it does well and what it misses.
3. Read the critiques of the previous draft. Note the specific suggestions and concerns that group members raised, and any new points of agreement or disagreement.
4. Plan the {num_statements} slates using the guidance above: list the agreed points, the sides of each disagreement with their supporters, and a one-line plan for each slate. Check that the plans are distinct.
5. Using the opinions, the previous draft, and the critiques, write {num_statements} revised funding slates. Check that every amount is $0 or at least the proposal's minimum and that each total is at most {pool}. If a slate leaves money unallocated, make sure there is a specific reason for it; otherwise use the rest of the fund. Check that there are exactly {num_statements} slates. Refer to opinion and critique numbers in your reasoning, but not in the slates themselves.

{_FORMAT.format(num_statements=num_statements)}

{example}


Below you will find the question,{context_description} the proposals, the individual opinions, the previous draft funding slate, and the critiques from the group members.


Question: {question}

{context_section}Proposals:
{funding.format_proposals(proposals)}

Individual Opinions:
"""
  for i, opinion in enumerate(opinions):
    prompt += f'Opinion Person {i+1}: {opinion}\n'

  prompt += f"""
Previous Draft Funding Slate:
{previous_winner}

Critiques of the Previous Draft:
"""

  for i, critique in enumerate(critiques):
    prompt += f'Critique Person {i+1}: {critique}\n'

  return prompt.strip()


def _generate_opinion_only_prompt(
    question: str,
    opinions: Sequence[str],
    num_statements: int,
    proposals: Sequence[funding.Proposal],
    funding_pool: int,
    context: str | Sequence[str | None] | None = None,
) -> str:
  """Generates a prompt for the LLM using only the opinions."""
  context_section = utils.format_context(context)
  data_description = (
      'the question, context about the participants, the proposals, and the'
      ' individual opinions'
      if context_section
      else 'the question, the proposals, and the individual opinions'
  )
  pool = funding.format_amount(funding_pool)
  example = _example(
      """1. Everyone supports the tool library (Opinions 1, 2, 3). Most people want the Hampi trip (Opinions 1, 3), and Opinion 2 wants yoga instead of the trip.
2. The slates should all fund the tool library. They should differ on whether to put most of the money into the trip or to fund yoga as well.
3. Slate 1 funds the trip above its minimum (Opinions 1, 3) and gives the tool library its ideal. Slate 2 gives the trip its minimum so that yoga can be funded too (Opinion 2). Both total $1,000.""")

  prompt = f"""
You are helping a group of people at Edge City decide how to split a community fund. {funding.SETTING}

{funding.fund_description(funding_pool)}

Each person in the group shared their opinion. Your job is to generate {num_statements} draft funding slates that capture what they agree on. A funding slate says how much money each proposal gets and why. The group will vote on these slates, so make them real alternatives that differ clearly from each other.

Base the slates on what the group members said. Use proposal details, such as cost and how realistic a project is, only to settle choices that the opinions leave open, and never claim support that no one expressed.

{_RULES.format(pool=pool)}

{_DIVERSITY.format(num_statements=num_statements, min_shift=_min_shift(funding_pool), extra="")}

Please think through this task step-by-step:

1. Read the individual opinions carefully. Note which proposals people support or oppose, where they agree, and where they disagree.
2. Plan the {num_statements} slates using the guidance above: list the agreed points, the sides of each disagreement with their supporters, and a one-line plan for each slate. Check that the plans are distinct.
3. Using your analysis, write {num_statements} draft funding slates. Check that every amount is $0 or at least the proposal's minimum and that each total is at most {pool}. If a slate leaves money unallocated, make sure there is a specific reason for it; otherwise use the rest of the fund. Check that there are exactly {num_statements} slates. Refer to opinion numbers in your reasoning, but not in the slates themselves.

{_FORMAT.format(num_statements=num_statements)}

{example}


Below you will find {data_description} of the group members.

Question: {question}

{context_section}Proposals:
{funding.format_proposals(proposals)}

Individual Opinions:
"""

  for i, opinion in enumerate(opinions):
    prompt += f'Opinion Person {i+1}: {opinion}\n'

  return prompt.strip()


def _generate_prompt(
    question: str,
    opinions: Sequence[str],
    num_statements: int,
    proposals: Sequence[funding.Proposal],
    funding_pool: int,
    previous_winner: str | None = None,
    critiques: Sequence[str] | None = None,
    context: str | Sequence[str | None] | None = None,
) -> str:
  """Generates a prompt for the LLM."""
  if previous_winner is None:
    return _generate_opinion_only_prompt(
        question, opinions, num_statements, proposals, funding_pool, context)
  else:
    return _generate_opinion_critique_prompt(
        question, opinions, previous_winner, critiques, num_statements,
        proposals, funding_pool, context,
    )


def _process_model_response(response: str) -> tuple[list[str], str]:
  """Processes the model's response, extracting the slates and explanation.

  Args:
      response: The raw model response.

  Returns:
      A tuple of (slates as JSON text, explanation). If the response format is
      incorrect, returns ([], "INCORRECT_TEMPLATE").
  """
  match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
  if match:
    parts = [part.strip() for part in match.group(1).split('<sep>')]
    if len(parts) > 1:
      return parts[1:], parts[0]
  return [], 'INCORRECT_TEMPLATE'


@dataclasses.dataclass(frozen=True)
class _Candidate:
  """A valid slate, with the side of the group it represents."""

  slate: funding.FundingSlate
  side: str
  explanation: str


def _side(slate_text: str) -> str:
  """Returns the normalized side label of a slate that parsed as JSON."""
  text = re.sub(r'^```(?:json)?\s*|\s*```$', '', slate_text.strip())
  side = json.loads(text).get('side')
  return ' '.join(side.lower().split()) if isinstance(side, str) else ''


def _distance(a: funding.FundingSlate, b: funding.FundingSlate) -> int:
  """Total dollars that differ between two slates over the same proposals."""
  return sum(
      abs(x.amount - y.amount) for x, y in zip(a.allocations, b.allocations))


def _select(
    candidates: Sequence[_Candidate], num_statements: int
) -> list[_Candidate]:
  """Picks up to num_statements distinct, spread-out slates.

  Repeated slates are dropped. Each side gets a share of the picks in
  proportion to how many slates the model wrote for it, and at least one, since
  the model writes more slates for views with more support. Within those
  shares, each pick is the slate farthest from the slates already picked.

  Args:
    candidates: Valid slates, in the order the model wrote them.
    num_statements: How many slates to pick.

  Returns:
    The picked slates, in the order the model wrote them.
  """
  unique = []
  for candidate in candidates:
    if all(_distance(candidate.slate, u.slate) for u in unique):
      unique.append(candidate)
  if len(unique) <= num_statements:
    return unique

  counts = {}
  for candidate in unique:
    counts[candidate.side] = counts.get(candidate.side, 0) + 1
  # With more sides than picks, the sides with the most slates get one each.
  sides = sorted(counts, key=lambda side: -counts[side])[:num_statements]
  quotas = {side: 1 for side in sides}
  while sum(quotas.values()) < num_statements:
    # Give the next pick to the side furthest below its proportional share.
    side = max(
        (side for side in sides if quotas[side] < counts[side]),
        key=lambda side: (
            counts[side] / len(unique) * num_statements - quotas[side]),
    )
    quotas[side] += 1

  picked = []
  while len(picked) < num_statements:
    taken = {}
    for i in picked:
      taken[unique[i].side] = taken.get(unique[i].side, 0) + 1
    eligible = [
        i for i in range(len(unique))
        if i not in picked
        and taken.get(unique[i].side, 0) < quotas.get(unique[i].side, 0)
    ]
    picked.append(max(
        eligible,
        key=lambda i: min(
            (_distance(unique[i].slate, unique[j].slate) for j in picked),
            default=0),
    ))
  return [unique[i] for i in sorted(picked)]


def _pad(
    selected: Sequence[_Candidate],
    num_statements: int,
    proposals: Sequence[funding.Proposal],
    funding_pool: int,
) -> list[_Candidate]:
  """Fills up to num_statements slates by spending less than the whole fund.

  When the rules leave fewer distinct slates than requested, each extra slate
  copies a selected one but gives one proposal only its minimum, leaving the
  difference unallocated. Proposals that every selected slate funds at the same
  amount are treated as agreed points and never cut. Slates take turns, and
  each cuts its proposal with the most money above the minimum first.

  Returns:
    The selected slates followed by the extra ones.
  """
  padded = list(selected)
  agreed = {
      k for k in range(len(proposals))
      if len({c.slate.allocations[k].amount for c in selected}) == 1
  }
  cuts = [
      sorted(
          (k for k, allocation in enumerate(candidate.slate.allocations)
           if k not in agreed
           and allocation.amount > proposals[k].minimum > 0),
          key=lambda k, c=candidate: (
              proposals[k].minimum - c.slate.allocations[k].amount),
      )
      for candidate in selected
  ]
  for depth in range(len(proposals)):
    for candidate, candidate_cuts in zip(selected, cuts):
      if len(padded) == num_statements:
        return padded
      if depth >= len(candidate_cuts):
        continue
      k = candidate_cuts[depth]
      proposal = proposals[k]
      old = candidate.slate.allocations[k]
      allocations = list(candidate.slate.allocations)
      allocations[k] = funding.Allocation(
          old.proposal_id, proposal.minimum,
          f'{old.reason} This version gives only the minimum instead of'
          f' {funding.format_amount(old.amount)}.')
      total = sum(allocation.amount for allocation in allocations)
      slate = funding.FundingSlate(
          f'{candidate.slate.summary} This version gives {proposal.title} only'
          f' its minimum and leaves'
          f' {funding.format_amount(funding_pool - total)} unallocated, to'
          ' give the group another option.',
          tuple(allocations),
      )
      if all(_distance(slate, p.slate) for p in padded):
        padded.append(_Candidate(slate, candidate.side, candidate.explanation))
  return padded


class FundingSlateModel(base_model.BaseStatementModel):
  """Chain-of-thought model that generates diverse funding slates.

  Each statement is a funding slate rendered as text: a summary, then the
  amount and reason for every proposal, then the total. Slates that use
  unknown proposal IDs, give a proposal less than its minimum (other than $0),
  or spend more than the funding pool are rejected and resampled.

  If the model writes more slates than requested, the most spread-out ones
  that still cover every side of the group's disagreements are kept. If it
  writes too few distinct slates, because the rules leave few options, the
  rest are filled by funding a proposal at its minimum and leaving the rest of
  the money unallocated.
  """

  def __init__(
      self,
      proposals: Sequence[funding.Proposal],
      funding_pool: int = funding.DEFAULT_FUNDING_POOL,
      oversample: float = 1.0,
  ):
    """Initializes the model.

    Args:
      proposals: The proposals that can receive funding.
      funding_pool: Total amount of money to distribute, in whole dollars.
      oversample: How many candidate slates to write per requested slate. For
        example, 1.5 picks 8 slates from 12 candidates. In evaluations this
        spread the slates out but covered the group's views less evenly.
    """
    funding.validate_proposals(proposals, funding_pool)
    if oversample < 1:
      raise ValueError(f'oversample must be at least 1, got {oversample}.')
    self._proposals = list(proposals)
    self._funding_pool = funding_pool
    self._oversample = oversample

  def generate_statement(
      self,
      llm_client: base_client.LLMClient,
      question: str,
      opinions: Sequence[str],
      previous_winner: str | None = None,
      critiques: Sequence[str] | None = None,
      context: str | Sequence[str | None] | None = None,
      seed: int | None = None,
      num_retries_on_error: int = 1,
  ) -> base_model.StatementResult:
    """Generates a slate (see base model)."""
    return self.generate_statements(
        llm_client,
        question,
        opinions,
        num_statements=1,
        previous_winner=previous_winner,
        critiques=critiques,
        context=context,
        seed=seed,
        num_retries_on_error=num_retries_on_error,
    )[0]

  def _sample_candidates(
      self,
      llm_client: base_client.LLMClient,
      question: str,
      opinions: Sequence[str],
      num_candidates: int,
      previous_winner: str | None,
      critiques: Sequence[str] | None,
      context: str | Sequence[str | None] | None,
      seed: int | None,
  ) -> tuple[list[_Candidate], list[str]]:
    """Asks the model for slates once. Returns (valid slates, errors)."""

    prompt = _generate_prompt(
        question, opinions, num_candidates, self._proposals,
        self._funding_pool, previous_winner, critiques, context)
    response = llm_client.sample_text(
        prompt,
        max_tokens=_MAX_TOKENS,
        terminators=['</answer>'],
        timeout=_TIMEOUT_SECONDS,
        seed=seed,
    )
    slates, explanation = _process_model_response(response)
    if not slates:
      return [], [explanation]
    candidates = []
    errors = []
    for slate_text in slates:
      try:
        slate = funding.parse_slate(
            slate_text, self._proposals, self._funding_pool)
      except ValueError as exc:
        errors.append(str(exc))
        continue
      candidates.append(_Candidate(slate, _side(slate_text), explanation))
    return candidates, errors

  def generate_statements(
      self,
      llm_client: base_client.LLMClient,
      question: str,
      opinions: Sequence[str],
      num_statements: int,
      previous_winner: str | None = None,
      critiques: Sequence[str] | None = None,
      context: str | Sequence[str | None] | None = None,
      seed: int | None = None,
      num_retries_on_error: int = 1,
  ) -> list[base_model.StatementResult]:
    """Generates up to num_statements valid slates (see base model).

    Invalid and repeated slates are dropped, and the model is asked again for
    the missing ones until it runs out of retries.

    Raises:
      ValueError: If no valid slate was generated after all retries.
    """
    if num_retries_on_error is None:
      num_retries_on_error = 0
    else:
      if num_retries_on_error < 0:
        raise ValueError('num_retries_on_error must be None or at least 0.')
    candidates = []
    errors = []
    selected = []
    for _ in range(num_retries_on_error + 1):
      missing = num_statements - len(selected)
      new_candidates, new_errors = self._sample_candidates(
          llm_client, question, opinions,
          math.ceil(missing * self._oversample), previous_winner, critiques,
          context, seed)
      candidates += new_candidates
      errors += new_errors
      selected = _select(candidates, num_statements)
      if len(selected) == num_statements:
        break
      if seed is not None:
        seed += 1
      print(
          f'Retrying for {num_statements - len(selected)} more slates.'
          f' Errors: {errors[-3:]}'
      )

    if not selected:
      raise ValueError(
          f'No valid funding slates after {num_retries_on_error + 1} attempts.'
          f' Last errors: {errors[-3:]}'
      )
    selected = _pad(
        selected, num_statements, self._proposals, self._funding_pool)
    return [
        base_model.StatementResult(
            funding.render_slate(
                candidate.slate, self._proposals, self._funding_pool),
            candidate.explanation,
        )
        for candidate in selected
    ]
