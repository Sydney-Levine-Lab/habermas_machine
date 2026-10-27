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


def _slate(p1, p2, side='Side'):
  return json.dumps({
      'side': side,
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
        num_statements=2, num_retries_on_error=0)
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

  def test_generate_statements_picks_spread_out_slates_covering_sides(self):
    model = funding_slate_model.FundingSlateModel(PROPOSALS, 1000)
    response = '<answer>R<sep>' + '<sep>'.join([
        _slate(100, 900, 'Trip'), _slate(100, 900, 'Trip'),
        _slate(150, 850, 'Trip'), _slate(300, 700, 'Both'),
        _slate(1000, 0, 'Cleanup'),
    ]) + '</answer>'
    results = model.generate_statements(
        mock_client.MockClient(response), 'Q?', ['O1'], num_statements=3,
        num_retries_on_error=0)
    p1_lines = [r.statement.split('\n')[2] for r in results]
    self.assertEqual(p1_lines, [
        '- P1 (Beach cleanup): $100. Reason.',
        '- P1 (Beach cleanup): $300. Reason.',
        '- P1 (Beach cleanup): $1,000. Reason.',
    ])

  def test_generate_statements_asks_for_oversampled_slates(self):
    client = _RecordingClient([f'<answer>R<sep>{_slate(300, 600)}</answer>'])
    model = funding_slate_model.FundingSlateModel(
        PROPOSALS, 1000, oversample=1.5)
    model.generate_statements(
        client, 'Q?', ['O1'], num_statements=4, num_retries_on_error=0)
    self.assertIn('exactly 6 slates', client.prompts[0])

  def test_generate_statements_pads_by_leaving_money_unallocated(self):
    model = funding_slate_model.FundingSlateModel(PROPOSALS, 1000)
    response = (
        f'<answer>R<sep>{_slate(100, 900)}<sep>{_slate(300, 700)}</answer>')
    results = model.generate_statements(
        mock_client.MockClient(response), 'Q?', ['O1'], num_statements=4,
        num_retries_on_error=0)
    self.assertLen(results, 4)
    self.assertIn('gives Hampi trip only its minimum and leaves $300'
                  ' unallocated', results[2].statement)
    self.assertIn('Total: $700 of $1,000', results[2].statement)
    self.assertIn('gives Beach cleanup only its minimum and leaves $200'
                  ' unallocated', results[3].statement)


class SelectTest(absltest.TestCase):

  def _candidate(self, p1, p2, side):
    slate = funding.parse_slate(_slate(p1, p2), PROPOSALS, 1000)
    return funding_slate_model._Candidate(slate, side, 'E')

  def test_keeps_every_side_before_spreading_out(self):
    candidates = [
        self._candidate(0, 900, 'trip'), self._candidate(0, 1000, 'trip'),
        self._candidate(100, 600, 'trip'), self._candidate(300, 700, 'both'),
    ]
    selected = funding_slate_model._select(candidates, 2)
    self.assertEqual([c.side for c in selected], ['trip', 'both'])

  def test_shares_picks_in_proportion_to_sides(self):
    candidates = [self._candidate(0, 600 + 50 * i, 'trip') for i in range(6)]
    candidates += [self._candidate(300, 600, 'both'),
                   self._candidate(300, 700, 'both')]
    selected = funding_slate_model._select(candidates, 4)
    self.assertEqual(
        [c.side for c in selected], ['trip', 'trip', 'trip', 'both'])

  def test_drops_repeated_slates(self):
    candidates = [self._candidate(0, 900, 'a'), self._candidate(0, 900, 'b')]
    self.assertLen(funding_slate_model._select(candidates, 2), 1)


class _RecordingClient(mock_client.MockClient):
  """Returns the given responses in turn and records the prompts."""

  def __init__(self, responses):
    super().__init__()
    self._responses = list(responses)
    self.prompts = []

  def sample_text(self, prompt, **kwargs):
    self.prompts.append(prompt)
    return self._responses[min(len(self.prompts), len(self._responses)) - 1]


if __name__ == '__main__':
  absltest.main()
