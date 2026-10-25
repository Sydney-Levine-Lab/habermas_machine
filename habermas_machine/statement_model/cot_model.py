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

"""A model that uses the chain-of-thought method to generate statements."""

from collections.abc import Sequence

import re

from habermas_machine.llm_client import base_client
from habermas_machine.statement_model import base_model


def _generate_opinion_critique_prompt(
    question: str,
    opinions: Sequence[str],
    previous_winner: str,
    critiques: Sequence[str],
) -> str:
  """Generates a prompt using opinions, previous winner, and critiques."""

  prompt = f"""You are helping a group of people at Edge City reach a shared decision about something that will happen in the village. Edge City is a gathering on the coast of Goa where people come to do their best work of the year together. Mornings start with exercise or time in nature. Afternoons are for coworking at The Circle, a hub that faces the beach. Evenings have fireside talks, workshops, demo days, and dinners that people organize themselves. Small groups also run programs throughout the event on topics such as AI, crypto, longevity, governance, and consciousness.

Each person in the group shared their opinion, someone wrote a first draft of a consensus statement, and group members then critiqued that draft. Your job is to write a revised consensus statement that uses this feedback to better reflect what the group wants. Make sure the revised statement does not conflict with any of the individual opinions.

Please think through this task step-by-step:

1. Read the individual opinions carefully. Note the main themes, where people agree, and where they disagree.
2. Review the previous draft consensus statement. Note what it does well and what it misses.
3. Read the critiques of the previous draft. Note the specific suggestions and concerns that group members raised.
4. Using the opinions, the previous draft, and the critiques, write a revised consensus statement that responds to these concerns and better reflects what the group wants. The statement should be clear and short, answer the question, and *not conflict* with any of the individual opinions. Refer to opinion and critique numbers when you explain your changes.

Provide your answer in the following format:
<answer>
[Your step-by-step reasoning and explanation for the revised statement]
<sep>
[Revised consensus statement]
</answer>

Example:
<answer>
1. Everyone wants a weekly community dinner (Opinions 1, 2, 3). People disagree on where to hold it (Opinions 2 and 3 prefer the beach, Opinion 1 prefers The Circle) and on how to handle food (Opinion 1 wants to hire a caterer, Opinion 3 wants a potluck).
2. The previous draft put the dinner at The Circle but didn't say how to handle food, which was the main concern in Critique 1.
3. Critique 1 asked for details on food, and Critique 2 suggested the beach, as Opinion 2 did.
4. The revised statement keeps the dinner at The Circle and combines a caterer for the main dishes with sides that people bring (Opinions 1 and 3, Critique 1). It adds a beach dinner once a month for those who prefer the beach (Opinion 2, Critique 2). The Circle stays the main location because it is easier to set up and works when it rains.
<sep>
We will hold a weekly community dinner at The Circle, since it is easier to set up and works when it rains. A caterer will provide the main dishes, and people can bring sides to share. Once a month, we will move the dinner to the beach.
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
) -> str:
  """Generates a prompt for the LLM using only the opinions."""
  prompt = f"""
You are helping a group of people at Edge City reach a shared decision about something that will happen in the village. Edge City is a gathering on the coast of Goa where people come to do their best work of the year together. Mornings start with exercise or time in nature. Afternoons are for coworking at The Circle, a hub that faces the beach. Evenings have fireside talks, workshops, demo days, and dinners that people organize themselves. Small groups also run programs throughout the event on topics such as AI, crypto, longevity, governance, and consciousness.

Each person in the group shared their opinion. Your job is to write a draft consensus statement that captures what they agree on and reflects what the group as a whole wants. The draft statement must not conflict with any of the individual opinions.

Please think through this task step-by-step:

1. Read the individual opinions carefully. Note the main themes, where people agree, and where they disagree.
2. Using your analysis, write a clear and short consensus statement that reflects the views the group members share. The statement should answer the question and *not conflict* with any of the individual opinions. Refer to opinion numbers to show how the draft reflects the group's views.

Provide your answer in the following format:
<answer>
[Your step-by-step reasoning and explanation for the statement]
<sep>
[Draft consensus statement]
</answer>

Example:
<answer>
1. Most opinions want a final demo day where anyone can present (Opinions 1, 3, 4) and want the demos kept short (Opinions 2, 4). Some also want time afterward to talk with presenters (Opinions 2, 3).
2. The draft statement calls for a demo day open to everyone, with short demos and time afterward to talk, which reflects the views that most opinions share.
<sep>
We will hold a final demo day where anyone can present. Demos will be short, and we will leave time afterward for people to talk with presenters about their work.
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
    previous_winner: str | None = None,
    critiques: Sequence[str] | None = None,
) -> str:
  """Generates a prompt for the LLM."""
  if previous_winner is None:
    return _generate_opinion_only_prompt(question, opinions)
  else:
    return _generate_opinion_critique_prompt(
        question, opinions, previous_winner, critiques
    )


def _process_model_response(response: str) -> tuple[str, str]:
  """Processes the model's response, extracting the statement and explanation.

  Args:
      response: The raw model response.

  Returns:
      A tuple of (statement, explanation).  If the response format is
      incorrect, returns ("", "INCORRECT_TEMPLATE").
  """
  match = re.search(
      r'<answer>\s*(.*?)\s*<sep>\s*(.*?)\s*</answer>', response, re.DOTALL
  )
  if match:
    explanation = match.group(1).strip()
    statement = match.group(2).strip()
    return statement, explanation
  else:
    return '', 'INCORRECT_TEMPLATE'


class COTModel(base_model.BaseStatementModel):
  """Statement model that uses chain-of-thought reasoning."""

  def generate_statement(
      self,
      llm_client: base_client.LLMClient,
      question: str,
      opinions: Sequence[str],
      previous_winner: str | None = None,
      critiques: Sequence[str] | None = None,
      seed: int | None = None,
      override_prompt: str | None = None,
      num_retries_on_error: int = 1,
  ) -> base_model.StatementResult:
    """Generates a statement (see base model)."""
    if num_retries_on_error is None:
      num_retries_on_error = 0
    else:
      if num_retries_on_error < 0:
        raise ValueError('num_retries_on_error must be None or at least 0.')
    prompt = _generate_prompt(question, opinions, previous_winner, critiques)
    statement, explanation = '', ''  # Dummy result.
    for _ in range(num_retries_on_error):
      response = llm_client.sample_text(
          prompt, terminators=['</answer>'], seed=seed)

      statement, explanation = _process_model_response(response)
      if len(statement) > 5 and 'INCORRECT' not in explanation:
        return base_model.StatementResult(statement, explanation)
      else:
        if seed is not None:
          seed += 1
          print(f'Retrying with new seed. Explanation: {explanation}')

    # If we reach here, all retries failed. Return the last result.
    return base_model.StatementResult(statement, explanation)


