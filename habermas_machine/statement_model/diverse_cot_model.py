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

"""A chain-of-thought model that generates diverse statements in one call."""

from collections.abc import Sequence

import re

from habermas_machine.llm_client import base_client
from habermas_machine.statement_model import base_model

# Generating many statements in one call needs a long response.
_MAX_TOKENS = 16384
_TIMEOUT_SECONDS = 300


def _generate_opinion_critique_prompt(
    question: str,
    opinions: Sequence[str],
    previous_winner: str,
    critiques: Sequence[str],
    num_statements: int,
) -> str:
  """Generates a prompt using opinions, previous winner, and critiques."""

  prompt = f"""You are assisting a citizens' jury in forming a consensus opinion on an important question. The jury members have provided their individual opinions, a first draft of a consensus statement was created, and critiques of that draft were gathered. Your role is to generate {num_statements} revised consensus statements that incorporate the feedback and aim to better represent the collective view of the jury. The jury will choose between the statements, so they should be genuinely different options that differ significantly from each other. All statements must be consistent on the points where the jury members agree. Where the jury members disagree, the statements should take different positions. If there is little disagreement, the statements should instead vary on other factors that plausibly matter for the question, but which the jury members did not mention explicitly.

Please think through this task step-by-step:

1. Carefully analyze the individual opinions, noting key themes, points of agreement, and areas of disagreement.
2. Review the previous draft consensus statement and identify its strengths and weaknesses.
3. Analyze the critiques of the previous draft, paying attention to specific suggestions and concerns raised by the jury members, and to any new points of agreement or disagreement.
4. Identify the dimensions on which the revised statements should differ: the areas of disagreement or, if there is little disagreement, other factors that plausibly matter for the question but were not mentioned explicitly.
5. Based on the opinions, the previous draft, and the critiques, write {num_statements} revised consensus statements. Each statement should be clear, concise, address the core issue posed in the question, and reflect all the points of agreement. The statements should differ significantly from each other on the dimensions identified in step 4. Refer to specific opinion and critique numbers in your reasoning, but not in the statements themselves.

Provide your answer in the following format, with exactly {num_statements} statements separated by <sep>:
<answer>
[Your step-by-step reasoning and explanation for the revised statements]
<sep>
[Revised consensus statement 1]
<sep>
[Revised consensus statement 2]
<sep>
...
<sep>
[Revised consensus statement {num_statements}]
</answer>

Example (with 3 statements):
<answer>
1. Opinions generally agree on the need for more green spaces (Opinions 1, 2, 3), but disagree on the specific location (Opinions 2 and 3 prefer the riverfront) and funding (Opinion 1 suggests a tax levy, Opinion 3 suggests private donations).
2. The previous draft suggested converting the old factory site into a park, but didn't address funding, which was a key concern in Critique 1.
3. Critiques highlighted the lack of funding details (Critique 1) and some preferred a different location (Critique 2 suggested the riverfront, echoing Opinion 2). All critiques accept that a new park should be created.
4. The revised statements should differ on location (factory site or riverfront) and funding (tax levy, private donations, or a combination).
5. Statement 1 keeps the factory site and funds it through a tax levy (Opinion 1, Critique 1). Statement 2 moves the park to the riverfront and funds it through private donations (Opinions 2 and 3, Critique 2). Statement 3 keeps the factory site as a lower-cost compromise and combines city funds with private donations (Opinions 1 and 3, Critique 1).
<sep>
We propose converting the old factory site into a park, funded by a dedicated tax levy so that funding is reliable and shared fairly by all residents.
<sep>
We propose creating a new park along the riverfront, funded by private donations so that no additional taxes are needed.
<sep>
We propose converting the old factory site into a park, as it avoids the higher costs of riverfront development, funded by a combination of city funds and private donations.
</answer>


Below you will find the question, the individual opinions, the previous draft consensus statement, and the critiques provided by the jury members.


Question: {question}

Individual Opinions:
"""
  for i, opinion in enumerate(opinions):
    prompt += f'Opinion Person {i+1}: {opinion}\n'

  prompt += f"""
Previous Draft Consensus Statement: {previous_winner}

Critiques of the Previous Draft:
"""

  for i, critique in enumerate(critiques):
    prompt += f'Critique Person {i+1}: {critique}\n'

  return prompt.strip()


def _generate_opinion_only_prompt(
    question: str,
    opinions: Sequence[str],
    num_statements: int,
) -> str:
  """Generates a prompt for the LLM using only the opinions."""
  prompt = f"""
You are assisting a citizens' jury in forming an initial consensus opinion on an important question. The jury members have provided their individual opinions. Your role is to generate {num_statements} draft consensus statements that capture the main points of agreement. The jury will choose between the statements, so they should be genuinely different options that differ significantly from each other. All statements must be consistent on the points where the jury members agree. Where the jury members disagree, the statements should take different positions. If there is little disagreement, the statements should instead vary on other factors that plausibly matter for the question, but which the jury members did not mention explicitly.

Please think through this task step-by-step:

1. Carefully analyze the individual opinions, noting key themes, points of agreement, and areas of disagreement.
2. Identify the dimensions on which the statements should differ: the areas of disagreement or, if there is little disagreement, other factors that plausibly matter for the question but were not mentioned explicitly.
3. Based on the analysis, write {num_statements} concise and clear draft consensus statements. Each statement should address the core issue posed in the question and reflect all the points of agreement. The statements should differ significantly from each other on the dimensions identified in step 2. Refer to specific opinion numbers in your reasoning, but not in the statements themselves.

Provide your answer in the following format, with exactly {num_statements} statements separated by <sep>:
<answer>
[Your step-by-step reasoning and explanation for the statements]
<sep>
[Draft consensus statement 1]
<sep>
[Draft consensus statement 2]
<sep>
...
<sep>
[Draft consensus statement {num_statements}]
</answer>

Example (with 3 statements):
<answer>
1. All opinions emphasize the importance of public transportation (Opinions 1, 2, 3, 4) and reducing car dependency (Opinions 2, 4). There is little disagreement.
2. Since there is little disagreement, the statements should differ on factors the opinions did not mention: whether to prioritize buses or rail, and how to pay for the investment.
3. Statement 1 prioritizes frequent bus services funded by a congestion charge, which also reduces car dependency (Opinions 2, 4). Statement 2 prioritizes expanding the rail network funded through general taxation. Statement 3 prioritizes making public transport free at the point of use to encourage people to switch from cars (Opinion 4).
<sep>
We believe that investing in public transport is crucial, starting with more frequent bus services, funded by a congestion charge that also discourages unnecessary car journeys.
<sep>
We believe that investing in public transport is crucial, with a focus on expanding the rail network, funded through general taxation as a long-term investment in the city.
<sep>
We believe that investing in public transport is crucial, and that it should be made free at the point of use so that as many people as possible choose it over the car.
</answer>


Below you will find the question and the individual opinions of the jury members.

Question: {question}

Individual Opinions:
"""

  for i, opinion in enumerate(opinions):
    prompt += f'Opinion Person {i+1}: {opinion}\n'

  return prompt.strip()


def _generate_prompt(
    question: str,
    opinions: Sequence[str],
    num_statements: int,
    previous_winner: str | None = None,
    critiques: Sequence[str] | None = None,
) -> str:
  """Generates a prompt for the LLM."""
  if previous_winner is None:
    return _generate_opinion_only_prompt(question, opinions, num_statements)
  else:
    return _generate_opinion_critique_prompt(
        question, opinions, previous_winner, critiques, num_statements
    )


def _process_model_response(response: str) -> tuple[list[str], str]:
  """Processes the model's response, extracting the statements and explanation.

  Args:
      response: The raw model response.

  Returns:
      A tuple of (statements, explanation).  If the response format is
      incorrect, returns ([], "INCORRECT_TEMPLATE").
  """
  match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
  if match:
    parts = [part.strip() for part in match.group(1).split('<sep>')]
    if len(parts) > 1:
      return parts[1:], parts[0]
  return [], 'INCORRECT_TEMPLATE'


class DiverseCOTModel(base_model.BaseStatementModel):
  """Chain-of-thought model that generates diverse statements in one call."""

  def generate_statement(
      self,
      llm_client: base_client.LLMClient,
      question: str,
      opinions: Sequence[str],
      previous_winner: str | None = None,
      critiques: Sequence[str] | None = None,
      seed: int | None = None,
      num_retries_on_error: int = 1,
  ) -> base_model.StatementResult:
    """Generates a statement (see base model)."""
    return self.generate_statements(
        llm_client,
        question,
        opinions,
        num_statements=1,
        previous_winner=previous_winner,
        critiques=critiques,
        seed=seed,
        num_retries_on_error=num_retries_on_error,
    )[0]

  def generate_statements(
      self,
      llm_client: base_client.LLMClient,
      question: str,
      opinions: Sequence[str],
      num_statements: int,
      previous_winner: str | None = None,
      critiques: Sequence[str] | None = None,
      seed: int | None = None,
      num_retries_on_error: int = 1,
  ) -> list[base_model.StatementResult]:
    """Generates exactly num_statements statements (see base model).

    If the model returns the wrong number of statements on every retry, the
    last result is truncated or padded with empty statements.
    """
    if num_retries_on_error is None:
      num_retries_on_error = 0
    else:
      if num_retries_on_error < 0:
        raise ValueError('num_retries_on_error must be None or at least 0.')
    prompt = _generate_prompt(
        question, opinions, num_statements, previous_winner, critiques)
    statements, explanation = [], ''  # Dummy result.
    for _ in range(num_retries_on_error):
      response = llm_client.sample_text(
          prompt,
          max_tokens=_MAX_TOKENS,
          terminators=['</answer>'],
          timeout=_TIMEOUT_SECONDS,
          seed=seed,
      )
      statements, explanation = _process_model_response(response)
      if len(statements) == num_statements and all(
          len(statement) > 5 for statement in statements
      ):
        break
      else:
        if 'INCORRECT' not in explanation:
          explanation = (
              f'INCORRECT_NUM_STATEMENTS ({len(statements)}): {explanation}')
        if seed is not None:
          seed += 1
          print(f'Retrying with new seed. Explanation: {explanation[:100]}')

    # If all retries failed, truncate or pad the last result.
    statements = statements[:num_statements]
    statements += [''] * (num_statements - len(statements))
    return [
        base_model.StatementResult(statement, explanation)
        for statement in statements
    ]
