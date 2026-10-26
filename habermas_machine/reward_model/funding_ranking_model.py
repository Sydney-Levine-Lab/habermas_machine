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

"""A chain-of-thought ranking model for funding slates."""

from collections.abc import Sequence

from typing_extensions import override

from habermas_machine import funding
from habermas_machine.reward_model import cot_ranking_model

_FORMAT = """Use Arrow notation for the ranking, where ">" means "preferred to". Ties are NOT allowed and items should be in descending order of preference so you can ONLY use ">" and the letters of the slates in the final ranking. Examples of valid final rankings: B > A, D > A > C > B, B > C > A > E > D.

Provide your answer in the following format:
<answer>
[Your step-by-step reasoning and explanation for the ranking]
<sep>
[Final ranking using arrow notation]
</answer>"""

_EXAMPLE = """For example, for three slates A, B and C a valid response could be:
<answer>
1. The participant most wants the Hampi trip and doesn't care much about yoga.
2. - Slate A funds the trip well above its minimum, which matches the participant's top priority.
   - Slate C funds the trip at its minimum, so it still happens but without a guide.
   - Slate B gives the trip $0 to fund yoga, which goes against the participant's opinion.
3. Based on this analysis, the ranking is: A > C > B
<sep>
A > C > B
</answer>

It is important to follow the template EXACTLY. So ALWAYS start with <answer>, then the explanation, then <sep> then only the final ranking and then </answer>."""


def _format_slates(statements: Sequence[str]) -> str:
  """Labels each slate with a letter."""
  return '\n\n'.join(
      f'Slate {chr(ord("A") + i)}:\n{statement.strip()}'
      for i, statement in enumerate(statements)
  )


def _generate_opinion_critique_prompt(
    question: str,
    opinion: str,
    statements: Sequence[str],
    previous_winner: str,
    critique: str,
    proposals: Sequence[funding.Proposal],
    funding_pool: int,
    group_context: str | None = None,
    participant_context: str | None = None,
) -> str:
  """Generates a prompt for the LLM using opinion and critique."""
  context_section, context_descriptions = cot_ranking_model._format_context(  # pylint: disable=protected-access
      group_context, participant_context)
  context_description = ''.join(
      f' {description},' for description in context_descriptions)
  return f"""As an AI assistant, your job is to rank funding slates in the order that a participant at Edge City would most likely prefer them, based on their opinion and their critique of the slate from a previous discussion round. {funding.SETTING}

{funding.fund_description(funding_pool)} A funding slate says how much money each proposal gets and why.

{_FORMAT}

Please think through this task step-by-step:

1. Analyze the participant's opinion, noting which proposals they support or oppose and how strongly.
2. Analyze the participant's critique of the slate from the previous round, noting what they want changed.
3. Compare each slate to the participant's opinion and critique, paying most attention to the amounts given to the proposals they care about.
4. Consider other parts of each slate that might appeal to or repel the participant, such as funding many small projects versus a few large ones, or leaving money unallocated.
5. Rank the slates accordingly using only ">" and the letters of the slates.

{_EXAMPLE}


Below you will find the question,{context_description} the proposals, the participant's opinion, the slate from the previous round, and the participant's critique of that slate. You will also find the slates to rank.

Question: {question}

{context_section}Proposals:
{funding.format_proposals(proposals)}

Participant's Opinion: {opinion}

Slate from previous round:
{previous_winner}

Critique: {critique}

Slates to rank:

{_format_slates(statements)}""".strip()


def _generate_opinion_only_prompt(
    question: str,
    opinion: str,
    statements: Sequence[str],
    proposals: Sequence[funding.Proposal],
    funding_pool: int,
    group_context: str | None = None,
    participant_context: str | None = None,
) -> str:
  """Generates a prompt for the LLM using only the opinion."""
  context_section, context_descriptions = cot_ranking_model._format_context(  # pylint: disable=protected-access
      group_context, participant_context)
  data_items = [
      'the question', *context_descriptions, 'the proposals',
      "the participant's opinion",
  ]
  data_description = ', '.join(data_items[:-1]) + ', and ' + data_items[-1]
  return f"""As an AI assistant, your job is to rank funding slates in the order that a participant at Edge City would most likely prefer them, based on their opinion. {funding.SETTING}

{funding.fund_description(funding_pool)} A funding slate says how much money each proposal gets and why.

{_FORMAT}

Please think through this task step-by-step:

1. Analyze the participant's opinion, noting which proposals they support or oppose and how strongly.
2. Compare each slate to the participant's opinion, paying most attention to the amounts given to the proposals they care about.
3. Consider other parts of each slate that might appeal to or repel the participant, such as funding many small projects versus a few large ones, or leaving money unallocated.
4. Rank the slates accordingly using only ">" and the letters of the slates.

{_EXAMPLE}


Below you will find {data_description}. You will also find the slates to rank.

Question: {question}

{context_section}Proposals:
{funding.format_proposals(proposals)}

Participant's Opinion: {opinion}

Slates to rank:

{_format_slates(statements)}""".strip()


class FundingRankingModel(cot_ranking_model.COTRankingModel):
  """Ranks funding slates using chain-of-thought reasoning."""

  def __init__(
      self,
      proposals: Sequence[funding.Proposal],
      funding_pool: int = funding.DEFAULT_FUNDING_POOL,
  ):
    """Initializes the model.

    Args:
      proposals: The proposals that can receive funding.
      funding_pool: Total amount of money to distribute, in whole dollars.
    """
    funding.validate_proposals(proposals, funding_pool)
    self._proposals = list(proposals)
    self._funding_pool = funding_pool

  @override
  def _generate_prompt(
      self,
      question: str,
      opinion: str,
      statements: Sequence[str],
      previous_winner: str | None = None,
      critique: str | None = None,
      group_context: str | None = None,
      participant_context: str | None = None,
  ) -> str:
    if previous_winner is None:
      return _generate_opinion_only_prompt(
          question, opinion, statements, self._proposals, self._funding_pool,
          group_context, participant_context,
      )
    return _generate_opinion_critique_prompt(
        question, opinion, statements, previous_winner, critique,
        self._proposals, self._funding_pool, group_context,
        participant_context,
    )
