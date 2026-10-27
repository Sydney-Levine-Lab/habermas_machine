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

import json

from absl.testing import absltest

from habermas_machine import funding
from habermas_machine.llm_client import mock_client
from habermas_machine.statement_model import funding_slate_model

PROPOSALS = [
    funding.Proposal('P1', 'Beach cleanup', 'Weekly cleanup.', 100, 300),
    funding.Proposal('P2', 'Hampi trip', 'Weekend trip.', 600, 900),
]


def _slate(p1, p2):
  return json.dumps({
      'summary': 'Summary.',
      'allocations': [
          {'id': 'P1', 'amount': p1, 'reason': 'Reason.'},
          {'id': 'P2', 'amount': p2, 'reason': 'Reason.'},
      ],
  })


class FundingSlateModelTest(absltest.TestCase):

  def test_opinion_only_prompt(self):
    prompt = funding_slate_model._generate_opinion_only_prompt(
        'How should we split the fund?', ['Fund the trip.', 'Fund cleanup.'],
        3, PROPOSALS, 2500)
    for text in ['How should we split the fund?', 'Fund the trip.',
                 'Fund cleanup.', 'Proposal P1: Beach cleanup',
                 'Minimum: $600', 'Ideal: $900', 'generate 3 draft funding',
                 'at most $2,500', 'fund of $2,500', 'move at least $250',
                 'Always write exactly 3 slates']:
      self.assertIn(text, prompt)

  def test_opinion_critique_prompt(self):
    prompt = funding_slate_model._generate_opinion_critique_prompt(
        'Question?', ['Opinion.'], 'Previous slate.', ['Critique.'], 2,
        PROPOSALS, 4000, context='Context.')
    for text in ['Previous slate.', 'Critique Person 1: Critique.',
                 'Context About the Participants: Context.',
                 'generate 2 revised funding slates', 'at most $4,000',
                 'move at least $400', 'stays close to it']:
      self.assertIn(text, prompt)

  def test_generate_statements_drops_invalid_slates(self):
    model = funding_slate_model.FundingSlateModel(PROPOSALS, 1000)
    response = (
        f'<answer>Reasoning.<sep>{_slate(100, 600)}<sep>{_slate(50, 600)}'
        f'<sep>{_slate(0, 900)}</answer>'
    )
    results = model.generate_statements(
        mock_client.MockClient(response), 'Q?', ['O1', 'O2'],
        num_statements=3, num_retries_on_error=0)
    self.assertLen(results, 2)
    self.assertIn('- P1 (Beach cleanup): $100.', results[0].statement)
    self.assertIn('Total: $900 of $1,000', results[1].statement)
    self.assertEqual(results[0].explanation, 'Reasoning.')

  def test_generate_statements_raises_without_valid_slates(self):
    model = funding_slate_model.FundingSlateModel(PROPOSALS, 1000)
    with self.assertRaisesRegex(ValueError, 'No valid funding slates'):
      model.generate_statements(
          mock_client.MockClient('Not an answer.'), 'Q?', ['O1'],
          num_statements=2, num_retries_on_error=1)

  def test_generate_statement(self):
    model = funding_slate_model.FundingSlateModel(PROPOSALS, 1000)
    result = model.generate_statement(
        mock_client.MockClient(f'<answer>R<sep>{_slate(300, 600)}</answer>'),
        'Q?', ['O1'])
    self.assertIn('Total: $900 of $1,000', result.statement)


if __name__ == '__main__':
  absltest.main()
