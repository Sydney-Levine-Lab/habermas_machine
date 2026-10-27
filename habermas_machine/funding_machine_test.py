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
from habermas_machine import funding_machine
from habermas_machine import machine
from habermas_machine import types
from habermas_machine.llm_client import mock_client
from habermas_machine.reward_model import funding_ranking_model
from habermas_machine.social_choice import utils as sc_utils
from habermas_machine.statement_model import funding_slate_model

PROPOSALS = [
    funding.Proposal('P1', 'Beach cleanup', 'Weekly cleanup.', 100, 300),
    funding.Proposal('P2', 'Hampi trip', 'Weekend trip.', 600, 900),
]


def _slate(p1, p2):
  return json.dumps({
      'side': 'Side',
      'summary': f'Give the cleanup ${p1}.',
      'allocations': [
          {'id': 'P1', 'amount': p1, 'reason': 'Reason.'},
          {'id': 'P2', 'amount': p2, 'reason': 'Reason.'},
      ],
  })


class _FundingClient(mock_client.MockClient):
  """Writes two slates when asked for slates, and ranks them otherwise."""

  def sample_text(self, prompt, **kwargs):
    if 'Slates to rank:' in prompt:
      return '<answer>R<sep>A > B</answer>'
    return f'<answer>R<sep>{_slate(100, 900)}<sep>{_slate(300, 700)}</answer>'


class MakeMachineTest(absltest.TestCase):

  def test_mediates_and_returns_winning_slate(self):
    client = _FundingClient()
    hm = funding_machine.make_machine(
        PROPOSALS, client, client, num_citizens=2, funding_pool=1000,
        num_candidates=2, seed=0)
    winner, candidates = hm.mediate(['Fund the trip.', 'Fund cleanup.'])
    self.assertLen(candidates, 2)
    slate = hm.statement_model.slate_for(winner)
    self.assertIsInstance(slate, funding.FundingSlate)
    self.assertEqual(slate.total, 1000)
    self.assertIsNone(hm.statement_model.slate_for('Unknown statement.'))

  def test_defaults(self):
    client = mock_client.MockClient()
    hm = funding_machine.make_machine(PROPOSALS, client, client, 3)
    self.assertEqual(hm._question, funding.default_question(4000))
    self.assertEqual(hm._num_candidates, 8)
    self.assertEqual(hm.statement_model.funding_pool, 4000)
    self.assertEqual(hm.reward_model.proposals, PROPOSALS)


class FundingModelCheckTest(absltest.TestCase):

  def _machine(self, statement_model, reward_model):
    return machine.HabermasMachine(
        question='Q?',
        statement_client=mock_client.MockClient(),
        reward_client=mock_client.MockClient(),
        statement_model=statement_model,
        reward_model=reward_model,
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            sc_utils.TieBreakingMethod.TBRC),
    )

  def test_rejects_funding_slates_with_text_ranking(self):
    with self.assertRaisesRegex(ValueError, 'must be used together'):
      self._machine(
          funding_slate_model.FundingSlateModel(PROPOSALS, 1000),
          types.RewardModel.CHAIN_OF_THOUGHT_RANKING.get_model())

  def test_rejects_text_statements_with_funding_ranking(self):
    with self.assertRaisesRegex(ValueError, 'must be used together'):
      self._machine(
          types.StatementModel.DIVERSE_CHAIN_OF_THOUGHT.get_model(),
          funding_ranking_model.FundingRankingModel(PROPOSALS, 1000))

  def test_rejects_different_pools(self):
    with self.assertRaisesRegex(ValueError, 'same proposals'):
      self._machine(
          funding_slate_model.FundingSlateModel(PROPOSALS, 1000),
          funding_ranking_model.FundingRankingModel(PROPOSALS, 2000))


if __name__ == '__main__':
  absltest.main()
