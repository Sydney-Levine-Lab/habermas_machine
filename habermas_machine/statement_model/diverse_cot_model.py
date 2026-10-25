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

  prompt = f"""You are helping a group of people at Edge City reach a shared decision about something that will happen in the village. Edge City is a gathering on the coast of Goa where people come to do their best work of the year together. Mornings start with exercise or time in nature. Afternoons are for coworking at The Circle, a hub that faces the beach. Evenings have fireside talks, workshops, demo days, and dinners that people organize themselves. Small groups also run programs throughout the event on topics such as AI, crypto, longevity, governance, and consciousness.

Each person in the group shared their opinion, someone wrote a first draft of a consensus statement, and group members then critiqued that draft. Your job is to generate {num_statements} revised consensus statements that use this feedback to better reflect what the group wants. The group will choose one of these statements, so make them real alternatives that differ clearly from each other. Every statement must match the points that all group members agree on. Where members disagree, the statements should take different sides. If members mostly agree, the statements should instead differ on other things that matter for the decision but that no one mentioned.

Please think through this task step-by-step:

1. Read the individual opinions carefully. Note the main themes, where people agree, and where they disagree.
2. Review the previous draft consensus statement. Note what it does well and what it misses.
3. Read the critiques of the previous draft. Note the specific suggestions and concerns that group members raised, and any new points of agreement or disagreement.
4. Decide how the revised statements should differ from each other: on the points of disagreement or, if members mostly agree, on other things that matter for the decision but that no one mentioned.
5. Using the opinions, the previous draft, and the critiques, write {num_statements} revised consensus statements. Each statement should be clear and short, answer the question, and include every point of agreement. The statements should differ clearly from each other in the ways you chose in step 4. Refer to opinion and critique numbers in your reasoning, but not in the statements themselves.

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
1. Everyone wants a weekly community dinner (Opinions 1, 2, 3). People disagree on where to hold it (Opinions 2 and 3 prefer the beach, Opinion 1 prefers The Circle) and on how to handle food (Opinion 1 wants to hire a caterer, Opinion 3 wants a potluck).
2. The previous draft put the dinner at The Circle but didn't say how to handle food, which was the main concern in Critique 1.
3. Critique 1 asked for details on food, and Critique 2 suggested the beach, as Opinion 2 did. All critiques accept a weekly dinner.
4. The revised statements should differ on location (The Circle or the beach) and food (caterer, potluck, or a mix).
5. Statement 1 keeps The Circle and hires a caterer (Opinion 1, Critique 1). Statement 2 moves the dinner to the beach and makes it a potluck (Opinions 2 and 3, Critique 2). Statement 3 keeps The Circle because it is easier to set up and works when it rains, and combines a caterer with dishes that people bring (Opinions 1 and 3, Critique 1).
<sep>
We will hold a weekly community dinner at The Circle and hire a caterer, paid for with a small fee from each person, so no one has to cook.
<sep>
We will hold a weekly community dinner on the beach as a potluck, so it costs nothing and people share food they made themselves.
<sep>
We will hold a weekly community dinner at The Circle, since it is easier to set up and works when it rains. A caterer will provide the main dishes, and people can bring sides to share.
</answer>


Below you will find the question, the individual opinions, the previous draft consensus statement, and the critiques from the group members.


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
You are helping a group of people at Edge City reach a shared decision about something that will happen in the village. Edge City is a gathering on the coast of Goa where people come to do their best work of the year together. Mornings start with exercise or time in nature. Afternoons are for coworking at The Circle, a hub that faces the beach. Evenings have fireside talks, workshops, demo days, and dinners that people organize themselves. Small groups also run programs throughout the event on topics such as AI, crypto, longevity, governance, and consciousness.

Each person in the group shared their opinion. Your job is to generate {num_statements} draft consensus statements that capture what they agree on. The group will choose one of these statements, so make them real alternatives that differ clearly from each other. Every statement must match the points that all group members agree on. Where members disagree, the statements should take different sides. If members mostly agree, the statements should instead differ on other things that matter for the decision but that no one mentioned.

Please think through this task step-by-step:

1. Read the individual opinions carefully. Note the main themes, where people agree, and where they disagree.
2. Decide how the statements should differ from each other: on the points of disagreement or, if members mostly agree, on other things that matter for the decision but that no one mentioned.
3. Using your analysis, write {num_statements} clear and short draft consensus statements. Each statement should answer the question and include every point of agreement. The statements should differ clearly from each other in the ways you chose in step 2. Refer to opinion numbers in your reasoning, but not in the statements themselves.

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
1. Everyone wants a final demo day where anyone can present (Opinions 1, 2, 3, 4), and several people want the demos kept short (Opinions 2, 4). Members mostly agree.
2. Since members mostly agree, the statements should differ on things the opinions did not mention: when to hold the demo day and what format to use.
3. Statement 1 holds short demos on one evening at The Circle, followed by a shared dinner. Statement 2 spreads short demos over two afternoons so more people can present without rushing (Opinion 4). Statement 3 replaces stage talks with short demos at tables around The Circle that people walk between (Opinion 2).
<sep>
We will hold a final demo day where anyone can present, with five-minute demos on one evening at The Circle, followed by a shared dinner.
<sep>
We will hold a final demo day where anyone can present, with short demos spread over two afternoons so more people can show their work without rushing.
<sep>
We will hold a final demo day where anyone can present. Instead of talks on a stage, presenters will give short demos at tables around The Circle, and people will walk between them.
</answer>


Below you will find the question and the individual opinions of the group members.

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
