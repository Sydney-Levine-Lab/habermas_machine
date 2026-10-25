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

"""Language Model that uses the OpenAI API."""

from collections.abc import Collection
import os

import openai
from typing_extensions import override

from habermas_machine.llm_client import base_client
from habermas_machine.llm_client import utils


# Optional sampling parameters that some models (e.g. reasoning models) reject.
# If the API rejects one of these, it is dropped for all future calls.
_OPTIONAL_PARAMS = ('temperature', 'stop', 'seed')


class OpenAIClient(base_client.LLMClient):
  """Language Model that uses the OpenAI Chat Completions API."""

  def __init__(
      self,
      model_name: str,
      *,
      api_key: str | None = None,
      base_url: str | None = None,
      reasoning_effort: str | None = None,
  ) -> None:
    """Initializes the instance.

    Args:
      model_name: which language model to use, e.g. 'gpt-6-luna'. For more
        details, see https://platform.openai.com/docs/models.
      api_key: OpenAI API key. Defaults to the OPENAI_API_KEY env variable.
      base_url: Optional base URL for OpenAI-compatible endpoints.
      reasoning_effort: Optional reasoning effort for reasoning models (e.g.
        'low', 'medium', 'high').
    """
    self._model_name = model_name
    self._reasoning_effort = reasoning_effort
    self._client = openai.OpenAI(
        api_key=api_key or os.environ['OPENAI_API_KEY'],
        base_url=base_url,
    )
    self._unsupported_params = set()

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = base_client.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = base_client.DEFAULT_TERMINATORS,
      temperature: float = base_client.DEFAULT_TEMPERATURE,
      timeout: float = base_client.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    optional_params = {'temperature': temperature}
    if terminators:
      # The API accepts at most 4 stop sequences. Truncation below handles the
      # rest (and models that do not support stop sequences at all).
      optional_params['stop'] = list(terminators)[:4]
    if seed is not None:
      # Cast in case of numpy ints, which are not JSON serializable.
      optional_params['seed'] = int(seed)

    while True:
      params = {
          k: v for k, v in optional_params.items()
          if k not in self._unsupported_params
      }
      if self._reasoning_effort is not None:
        params['reasoning_effort'] = self._reasoning_effort
      try:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{'role': 'user', 'content': prompt}],
            max_completion_tokens=max_tokens,
            timeout=timeout,
            **params,
        )
        break
      except openai.BadRequestError as e:
        param = (e.body or {}).get('param') if isinstance(e.body, dict) else None
        if param in _OPTIONAL_PARAMS and param not in self._unsupported_params:
          self._unsupported_params.add(param)
          continue
        raise
      except openai.APITimeoutError as e:
        raise TimeoutError(str(e)) from e

    text = response.choices[0].message.content or ''
    return utils.truncate(text, delimiters=terminators)
