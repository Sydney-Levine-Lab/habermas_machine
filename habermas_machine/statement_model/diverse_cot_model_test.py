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

from habermas_machine.llm_client import mock_client
from habermas_machine.statement_model import diverse_cot_model


class DiverseCOTModelTest(parameterized.TestCase):

  def test_generate_opinion_only_prompt(self):
    """Tests the _generate_opinion_only_prompt method."""
    question = "What is the best way to improve public transport?"
    opinions = ["Invest in more buses.", "Reduce fares."]
    prompt = diverse_cot_model._generate_opinion_only_prompt(
        question, opinions, 7)
    self.assertIn(question, prompt)
    for opinion in opinions:
      self.assertIn(opinion, prompt)
    self.assertIn("generate 7 draft consensus statements", prompt)
    self.assertIn("<sep>", prompt)

  def test_generate_opinion_critique_prompt(self):
    """Tests the _generate_opinion_critique_prompt method."""
    question = "How can we make our city greener?"
    opinions = ["Plant more trees.", "Create more parks."]
    previous_winner = "We should plant more trees in the city center."
    critiques = ["Planting trees is not enough.", "We need more parks."]
    prompt = diverse_cot_model._generate_opinion_critique_prompt(
        question, opinions, previous_winner, critiques, 7
    )
    self.assertIn(question, prompt)
    for text in [*opinions, previous_winner, *critiques]:
      self.assertIn(text, prompt)
    self.assertIn("generate 7 revised consensus statements", prompt)

  def test_process_model_response(self):
    """Tests the _process_model_response method."""
    response = "<answer>Explanation.\n<sep>\nFirst.\n<sep>\nSecond.</answer>"
    statements, explanation = diverse_cot_model._process_model_response(
        response)
    self.assertEqual(statements, ["First.", "Second."])
    self.assertEqual(explanation, "Explanation.")

    statements, explanation = diverse_cot_model._process_model_response(
        "This is just some text.")
    self.assertEqual(statements, [])
    self.assertEqual(explanation, "INCORRECT_TEMPLATE")

  @parameterized.parameters(1, 2, 3)
  def test_generate_statements_pads_or_truncates(self, num_statements):
    """Tests that generate_statements returns exactly num_statements."""
    model = diverse_cot_model.DiverseCOTModel()
    llm_client = mock_client.MockClient(
        response="<answer>Mock<sep>Statement A<sep>Statement B</answer>")
    results = model.generate_statements(
        llm_client, "Test question?", ["Opinion 1", "Opinion 2"],
        num_statements=num_statements,
    )
    self.assertLen(results, num_statements)
    expected = ["Statement A", "Statement B", ""][:num_statements]
    self.assertEqual([r.statement for r in results], expected)

  def test_generate_statement(self):
    """Tests the generate_statement method."""
    model = diverse_cot_model.DiverseCOTModel()
    llm_client = mock_client.MockClient(
        response="<answer>Mock explanation<sep>Mock statement</answer>")
    statement, explanation = model.generate_statement(
        llm_client, "Test question?", ["Opinion 1", "Opinion 2"]
    )
    self.assertEqual(statement, "Mock statement")
    self.assertEqual(explanation, "Mock explanation")


if __name__ == "__main__":
  absltest.main()
