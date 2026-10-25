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
from absl.testing import parameterized

from habermas_machine import machine
from habermas_machine import types
from habermas_machine.reward_model import mock_ranking_model
from habermas_machine.social_choice import utils as sc_utils
from habermas_machine.statement_model import mock_statement_model


class HabermasMachineTest(parameterized.TestCase):

  def test_habermas_machine(self):
    """Tests the habermas machine."""
    num_citizens = 2
    num_candidates = 3  # Generate only one candidate for easier checking
    question = 'What is the meaning of life?'
    opinions = ['42', 'To be happy.']
    critiques = ['I disagree.', 'I agree.']
    hm = machine.HabermasMachine(
        question=question,
        statement_client=types.LLMCLient.MOCK.get_client('mock_url'),
        reward_client=types.LLMCLient.MOCK.get_client('mock_url'),
        statement_model=types.StatementModel.MOCK.get_model(),
        reward_model=types.RewardModel.MOCK.get_model(),
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
        ),
        num_candidates=num_candidates,
        num_citizens=num_citizens,
        seed=0,
    )

    # Test initial state.
    self.assertEqual(hm._round, 0)
    self.assertEmpty(hm._opinions)
    self.assertEmpty(hm._critiques)
    self.assertEmpty(hm._previous_winners)
    self.assertEmpty(hm._previous_candidates)
    self.assertEqual(hm._num_citizens, num_citizens)
    self.assertEqual(hm._num_candidates, num_candidates)
    self.assertEqual(hm._verbose, False)
    self.assertEqual(hm._question, question)

    # Test overwrite previous winner.
    with self.assertRaises(ValueError):
      hm.overwrite_previous_winner('winner1')

    # Test opinion round.
    winner, sorted_statements = hm.mediate(opinions)

    self.assertEqual(hm._round, 1)
    self.assertEqual(hm._question, question)
    self.assertSequenceEqual(hm._opinions, opinions)
    self.assertEmpty(hm._critiques)
    self.assertSequenceEqual(hm._previous_winners, [winner])
    self.assertSequenceEqual(hm._previous_candidates, [sorted_statements])
    self.assertIn(winner, sorted_statements)
    for statement in sorted_statements:
      self.assertIn(question, statement)
      for opinion in opinions:
        self.assertIn(opinion, statement)

    # Test critique round.
    winner2, sorted_statements2 = hm.mediate(critiques)

    self.assertEqual(hm._round, 2)
    self.assertEqual(hm._question, question)
    self.assertSequenceEqual(hm._opinions, opinions)
    self.assertSequenceEqual(hm._critiques, [critiques])
    self.assertSequenceEqual(hm._previous_winners, [winner, winner2])
    self.assertSequenceEqual(
        hm._previous_candidates, [sorted_statements, sorted_statements2]
    )
    self.assertIn(winner2, sorted_statements2)
    for statement in sorted_statements2:
      self.assertIn(question, statement)
      for opinion in opinions:
        self.assertIn(opinion, statement)
      for critique in critiques:
        self.assertIn(critique, statement)

    # Test overwrite previous winner.
    hm.overwrite_previous_winner('winner3')
    self.assertEqual(hm._previous_winners[-1], 'winner3')

  def test_wrong_number_of_opinions(self):
    num_citizens = 2
    hm = machine.HabermasMachine(
        question='Question?',
        statement_client=types.LLMCLient.MOCK.get_client('mock_url'),
        reward_client=types.LLMCLient.MOCK.get_client('mock_url'),
        statement_model=types.StatementModel.MOCK.get_model(),
        reward_model=types.RewardModel.MOCK.get_model(),
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
        ),
        num_citizens=num_citizens,
        seed=0,
    )
    with self.assertRaises(ValueError):
      _, _ = hm.mediate(['opinion1'])

  def test_wrong_number_of_critiques(self):
    num_citizens = 2
    hm = machine.HabermasMachine(
        question='Question?',
        statement_client=types.LLMCLient.MOCK.get_client('mock_url'),
        reward_client=types.LLMCLient.MOCK.get_client('mock_url'),
        statement_model=types.StatementModel.MOCK.get_model(),
        reward_model=types.RewardModel.MOCK.get_model(),
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
        ),
        num_citizens=num_citizens,
        seed=0,
    )
    _, _ = hm.mediate(['opinion1', 'opinion2'])  # Opinion round.
    with self.assertRaises(ValueError):
      hm.mediate(['critique1', 'critique2', 'critique3'])

  def _make_machine(self, statement_model=None, reward_model=None, **kwargs):
    return machine.HabermasMachine(
        question='Question?',
        statement_client=types.LLMCLient.MOCK.get_client('mock_url'),
        reward_client=types.LLMCLient.MOCK.get_client('mock_url'),
        statement_model=(
            statement_model or types.StatementModel.MOCK.get_model()),
        reward_model=reward_model or types.RewardModel.MOCK.get_model(),
        social_choice_method=types.RankAggregation.SCHULZE.get_method(
            tie_breaking_method=sc_utils.TieBreakingMethod.TIES_ALLOWED
        ),
        num_candidates=2,
        **kwargs,
    )

  @parameterized.parameters('statement_context', 'ranking_context')
  def test_wrong_number_of_context_entries(self, name):
    with self.assertRaises(ValueError):
      self._make_machine(num_citizens=2, **{name: ['a', 'b', 'c']})

  def test_shared_context(self):
    ranking_calls = []

    class RecordingRankingModel(mock_ranking_model.MockRankingModel):

      def predict_ranking(self, *args, **kwargs):
        ranking_calls.append(
            (kwargs['group_context'], kwargs['participant_context']))
        return super().predict_ranking(*args, **kwargs)

    hm = self._make_machine(
        reward_model=RecordingRankingModel(),
        num_citizens=2,
        statement_context='Statement context.',
        ranking_context='Ranking context.',
    )
    winner, _ = hm.mediate(['opinion1', 'opinion2'])
    self.assertIn('Statement context.', winner)
    self.assertEqual(ranking_calls, [('Ranking context.', None)] * 2)

  def test_per_citizen_context_follows_citizens(self):
    """Tests that per-citizen context stays aligned with each citizen."""
    statement_calls = []
    ranking_calls = []

    class RecordingStatementModel(mock_statement_model.MockStatementModel):

      def generate_statement(self, *args, **kwargs):
        statement_calls.append((kwargs['opinions'], kwargs['context']))
        return super().generate_statement(*args, **kwargs)

    class RecordingRankingModel(mock_ranking_model.MockRankingModel):

      def predict_ranking(self, *args, **kwargs):
        ranking_calls.append((
            kwargs['opinion'],
            kwargs['group_context'],
            kwargs['participant_context'],
        ))
        return super().predict_ranking(*args, **kwargs)

    num_citizens = 5
    opinions = [f'opinion{i}' for i in range(num_citizens)]
    hm = self._make_machine(
        statement_model=RecordingStatementModel(),
        reward_model=RecordingRankingModel(),
        num_citizens=num_citizens,
        statement_context=[f'statement_context{i}' for i in range(5)],
        ranking_context=[f'ranking_context{i}' for i in range(5)],
    )
    hm.mediate(opinions)

    self.assertNotEmpty(statement_calls)
    for shuffled_opinions, context in statement_calls:
      self.assertEqual(
          [o.replace('opinion', 'statement_context')
           for o in shuffled_opinions],
          context,
      )
    self.assertLen(ranking_calls, num_citizens)
    for opinion, group_context, participant_context in ranking_calls:
      self.assertIsNone(group_context)
      self.assertEqual(
          opinion.replace('opinion', 'ranking_context'), participant_context)


if __name__ == '__main__':
  absltest.main()
