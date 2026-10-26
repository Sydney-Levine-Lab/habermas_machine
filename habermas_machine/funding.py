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

"""Proposals and funding slates for deciding how to split a community fund."""

from collections.abc import Sequence
import dataclasses
import json
import re

DEFAULT_FUNDING_POOL = 4000

SETTING = (
    'Edge City is a gathering on the coast of Goa where people come to do'
    ' their best work of the year together. Mornings start with exercise or'
    ' time in nature. Afternoons are for coworking at The Circle, a hub that'
    ' faces the beach. Evenings have fireside talks, workshops, demo days, and'
    ' dinners that people organize themselves. Small groups also run programs'
    ' throughout the event on topics such as AI, crypto, longevity,'
    ' governance, and consciousness.'
)


@dataclasses.dataclass(frozen=True)
class Proposal:
  """A project that attendees proposed for funding.

  Attributes:
    id: Unique identifier that slates use to refer to the proposal.
    title: Short title of the project.
    description: Details about the project.
    minimum: Smallest amount the project can happen with. A slate gives the
      project either $0 or at least this amount.
    ideal: Amount the project would ideally get. Slates may give more if they
      explain why.
  """

  id: str
  title: str
  description: str
  minimum: int
  ideal: int


@dataclasses.dataclass(frozen=True)
class Allocation:
  """Money that a slate gives to one proposal."""

  proposal_id: str
  amount: int
  reason: str


@dataclasses.dataclass(frozen=True)
class FundingSlate:
  """How much money each proposal gets and why."""

  summary: str
  allocations: tuple[Allocation, ...]

  @property
  def total(self) -> int:
    return sum(allocation.amount for allocation in self.allocations)


def format_amount(amount: int) -> str:
  """Formats a dollar amount, e.g. 4000 -> '$4,000'."""
  return f'${amount:,}'


def default_question(funding_pool: int = DEFAULT_FUNDING_POOL) -> str:
  """Returns the default question for a funding decision."""
  return (
      f'How should we distribute the {format_amount(funding_pool)} community'
      ' fund among the proposed projects?'
  )


def fund_description(funding_pool: int) -> str:
  """Describes the community fund and the proposals for prompts."""
  return (
      f'The community has a fund of {format_amount(funding_pool)} to give to'
      ' projects that attendees proposed to make community life at Edge City'
      ' better, more interesting, more useful, or more fun: a weekend trip, a'
      ' community experiment, a shared tool, a workshop, an event, something'
      ' to build, or a problem to solve. Proposals can be rough ideas rather'
      ' than polished grant applications. Each proposal has an ID, a minimum'
      ' amount it needs to happen, and an ideal amount.'
  )


def validate_proposals(
    proposals: Sequence[Proposal], funding_pool: int) -> None:
  """Raises ValueError if the proposals or funding pool are invalid."""
  if funding_pool <= 0:
    raise ValueError(f'funding_pool must be positive, got {funding_pool}.')
  if not proposals:
    raise ValueError('There should be at least one proposal.')
  ids = [proposal.id for proposal in proposals]
  if len(ids) != len(set(ids)):
    raise ValueError(f'Proposal IDs must be unique, got {ids}.')
  for proposal in proposals:
    if not proposal.id.strip():
      raise ValueError('Proposal IDs must not be empty.')
    if proposal.minimum < 0:
      raise ValueError(
          f'Proposal {proposal.id} has a negative minimum: {proposal.minimum}.')
    if proposal.ideal < proposal.minimum:
      raise ValueError(
          f'Proposal {proposal.id} has an ideal amount ({proposal.ideal})'
          f' below its minimum ({proposal.minimum}).'
      )


def format_proposals(proposals: Sequence[Proposal]) -> str:
  """Formats the proposals as a prompt section."""
  return '\n\n'.join(
      f'Proposal {proposal.id}: {proposal.title}\n'
      f'Minimum: {format_amount(proposal.minimum)}\n'
      f'Ideal: {format_amount(proposal.ideal)}\n'
      f'Description: {proposal.description}'
      for proposal in proposals
  )


def _parse_amount(value: object, proposal_id: str) -> int:
  """Parses a whole-dollar amount from JSON."""
  if isinstance(value, bool):
    raise ValueError(f'Amount for {proposal_id} is not a number: {value}.')
  if isinstance(value, float) and value.is_integer():
    value = int(value)
  if not isinstance(value, int):
    raise ValueError(
        f'Amount for {proposal_id} is not a whole-dollar number: {value}.')
  return value


def parse_slate(
    text: str,
    proposals: Sequence[Proposal],
    funding_pool: int,
) -> FundingSlate:
  """Parses and validates a funding slate written as JSON.

  Args:
    text: JSON object with a "summary" string and an "allocations" list of
      {"id", "amount", "reason"} objects. May be wrapped in a code fence.
    proposals: The proposals that can receive funding.
    funding_pool: Total amount of money available.

  Returns:
    The slate, with allocations in the same order as the proposals.

  Raises:
    ValueError: If the slate is malformed, does not cover every proposal
      exactly once, gives a proposal more than $0 but less than its minimum,
      or spends more than the funding pool.
  """
  text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
  try:
    data = json.loads(text)
  except json.JSONDecodeError as exc:
    raise ValueError(f'Slate is not valid JSON: {exc}') from exc
  if not isinstance(data, dict):
    raise ValueError('Slate is not a JSON object.')
  summary = data.get('summary')
  items = data.get('allocations')
  if not isinstance(summary, str) or not summary.strip():
    raise ValueError('Slate has no summary.')
  if not isinstance(items, list):
    raise ValueError('Slate has no allocations list.')

  proposals_by_id = {proposal.id: proposal for proposal in proposals}
  allocations_by_id = {}
  for item in items:
    if not isinstance(item, dict):
      raise ValueError(f'Allocation is not a JSON object: {item}.')
    proposal_id = item.get('id')
    if proposal_id not in proposals_by_id:
      raise ValueError(f'Unknown proposal ID: {proposal_id}.')
    if proposal_id in allocations_by_id:
      raise ValueError(f'Proposal {proposal_id} appears more than once.')
    amount = _parse_amount(item.get('amount'), proposal_id)
    minimum = proposals_by_id[proposal_id].minimum
    if amount < 0:
      raise ValueError(f'Amount for {proposal_id} is negative: {amount}.')
    if 0 < amount < minimum:
      raise ValueError(
          f'Amount for {proposal_id} ({format_amount(amount)}) is below its'
          f' minimum ({format_amount(minimum)}).'
      )
    reason = item.get('reason')
    if not isinstance(reason, str) or not reason.strip():
      raise ValueError(f'Allocation for {proposal_id} has no reason.')
    allocations_by_id[proposal_id] = Allocation(
        proposal_id, amount, reason.strip())

  missing = [id_ for id_ in proposals_by_id if id_ not in allocations_by_id]
  if missing:
    raise ValueError(f'Slate is missing proposals: {missing}.')
  slate = FundingSlate(
      summary.strip(),
      tuple(allocations_by_id[proposal.id] for proposal in proposals),
  )
  if slate.total > funding_pool:
    raise ValueError(
        f'Slate total ({format_amount(slate.total)}) is more than the funding'
        f' pool ({format_amount(funding_pool)}).'
    )
  return slate


def render_slate(
    slate: FundingSlate,
    proposals: Sequence[Proposal],
    funding_pool: int,
) -> str:
  """Renders a slate as text for participants, using the proposals' titles."""
  titles = {proposal.id: proposal.title for proposal in proposals}
  lines = [slate.summary, '']
  for allocation in slate.allocations:
    lines.append(
        f'- {allocation.proposal_id} ({titles[allocation.proposal_id]}):'
        f' {format_amount(allocation.amount)}. {allocation.reason}'
    )
  total = (
      f'Total: {format_amount(slate.total)} of {format_amount(funding_pool)}')
  if slate.total < funding_pool:
    total += f' ({format_amount(funding_pool - slate.total)} unallocated)'
  lines += ['', total]
  return '\n'.join(lines)
