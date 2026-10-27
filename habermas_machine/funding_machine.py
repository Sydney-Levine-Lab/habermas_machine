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

"""Creates a Habermas Machine that decides how to split a community fund."""

from collections.abc import Sequence
from typing import Any

from habermas_machine import funding
from habermas_machine import machine
from habermas_machine import types
from habermas_machine.llm_client import base_client
from habermas_machine.reward_model import funding_ranking_model
from habermas_machine.social_choice import base_method as base_social_choice
from habermas_machine.statement_model import funding_slate_model

# The slate prompts were tuned and evaluated with 8 slates per round.
DEFAULT_NUM_CANDIDATES = 8


def make_machine(
    proposals: Sequence[funding.Proposal],
    statement_client: base_client.LLMClient,
    reward_client: base_client.LLMClient,
    num_citizens: int,
    funding_pool: int = funding.DEFAULT_FUNDING_POOL,
    question: str | None = None,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    social_choice_method: base_social_choice.Base | None = None,
    oversample: float = 1.0,
    **kwargs: Any,
) -> machine.HabermasMachine:
  """Creates a Habermas Machine whose statements are funding slates.

  Uses FundingSlateModel and FundingRankingModel with the same proposals and
  pool. Call `machine.statement_model.slate_for(winner)` to get the winning
  FundingSlate back from its text.

  Args:
    proposals: The proposals that can receive funding.
    statement_client: LLM client used by the statement model.
    reward_client: LLM client used by the reward model.
    num_citizens: Number of citizens in the deliberation.
    funding_pool: Total amount of money to distribute, in whole dollars.
    question: Question to be answered. Defaults to
      `funding.default_question(funding_pool)`.
    num_candidates: Number of candidate slates to generate per round.
    social_choice_method: Method that aggregates the rankings. Defaults to
      Schulze with TBRC tie-breaking.
    oversample: Passed to FundingSlateModel.
    **kwargs: Other arguments for HabermasMachine, such as seed, verbose,
      num_retries_on_error, statement_context and ranking_context.

  Returns:
    The Habermas Machine.
  """
  if question is None:
    question = funding.default_question(funding_pool)
  if social_choice_method is None:
    social_choice_method = types.RankAggregation.SCHULZE.get_method(
        types.TieBreakingMethod.TBRC)
  return machine.HabermasMachine(
      question=question,
      statement_client=statement_client,
      reward_client=reward_client,
      statement_model=funding_slate_model.FundingSlateModel(
          proposals, funding_pool, oversample=oversample),
      reward_model=funding_ranking_model.FundingRankingModel(
          proposals, funding_pool),
      social_choice_method=social_choice_method,
      num_candidates=num_candidates,
      num_citizens=num_citizens,
      **kwargs,
  )
