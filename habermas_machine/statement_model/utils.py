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

"""Statement model utils."""

from collections.abc import Sequence


def format_context(context: str | Sequence[str | None] | None) -> str:
  """Formats context about the participants as a prompt section.

  Args:
    context: Either a single string about the group or one optional string per
      participant, in the same order as the opinions.

  Returns:
    The prompt section followed by a blank line, or an empty string if there is
    no context.
  """
  if not context:
    return ''
  if isinstance(context, str):
    return f'Context About the Participants: {context}\n\n'
  lines = [
      f'Context Person {i+1}: {person_context}\n'
      for i, person_context in enumerate(context)
      if person_context
  ]
  if not lines:
    return ''
  return 'Context About the Participants:\n' + ''.join(lines) + '\n'
