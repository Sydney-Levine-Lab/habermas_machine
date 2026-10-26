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

from absl.testing import absltest
import numpy as np

from habermas_machine import funding
from habermas_machine.llm_client import mock_client
from habermas_machine.reward_model import funding_ranking_model

PROPOSALS = [
    funding.Proposal('P1', 'Beach cleanup', 'Weekly cleanup.', 100, 300),
    funding.Proposal('P2', 'Hampi trip', 'Weekend trip.', 600, 900),
]


class FundingRankingModelTest(absltest.TestCase):

  def test_opinion_only_prompt(self):
    model = funding_ranking_model.FundingRankingModel(PROPOSALS, 2500)
    prompt = model._generate_prompt(
        'Question?', 'I want the trip.', ['Slate one.', 'Slate two.'],
        group_context='Group.')
    for text in ['I want the trip.', 'Slate A:\nSlate one.',
                 'Slate B:\nSlate two.', 'Proposal P2: Hampi trip',
                 'fund of $2,500', 'Context About the Participants: Group.']:
      self.assertIn(text, prompt)

  def test_opinion_critique_prompt(self):
    model = funding_ranking_model.FundingRankingModel(PROPOSALS)
    prompt = model._generate_prompt(
        'Question?', 'Opinion.', ['One.', 'Two.'],
        previous_winner='Previous.', critique='Critique.')
    for text in ['Slate from previous round:\nPrevious.',
                 'Critique: Critique.', 'fund of $4,000']:
      self.assertIn(text, prompt)

  def test_predict_ranking(self):
    model = funding_ranking_model.FundingRankingModel(PROPOSALS)
    result = model.predict_ranking(
        mock_client.MockClient('<answer>Reasoning.<sep>B > A</answer>'),
        'Question?', 'Opinion.', ['One.', 'Two.'])
    np.testing.assert_array_equal(result.ranking, [1, 0])


if __name__ == '__main__':
  absltest.main()
