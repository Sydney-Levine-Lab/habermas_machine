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
from absl.testing import parameterized

from habermas_machine import funding

PROPOSALS = [
    funding.Proposal('P1', 'Beach cleanup', 'Weekly cleanup.', 100, 300),
    funding.Proposal('P2', 'Hampi trip', 'Weekend trip.', 600, 900),
]


def _slate(allocations, summary='Summary.'):
  return json.dumps({
      'summary': summary,
      'allocations': [
          {'id': id_, 'amount': amount, 'reason': f'Reason {id_}.'}
          for id_, amount in allocations
      ],
  })


class FundingTest(parameterized.TestCase):

  def test_parse_slate_orders_by_proposals(self):
    slate = funding.parse_slate(
        _slate([('P2', 700), ('P1', 0)]), PROPOSALS, 1000)
    self.assertEqual(
        [(a.proposal_id, a.amount) for a in slate.allocations],
        [('P1', 0), ('P2', 700)],
    )
    self.assertEqual(slate.total, 700)

  def test_parse_slate_accepts_code_fence(self):
    text = '```json\n' + _slate([('P1', 100), ('P2', 900)]) + '\n```'
    self.assertEqual(funding.parse_slate(text, PROPOSALS, 1000).total, 1000)

  def test_parse_slate_allows_more_than_ideal(self):
    slate = funding.parse_slate(
        _slate([('P1', 400), ('P2', 0)]), PROPOSALS, 1000)
    self.assertEqual(slate.total, 400)

  @parameterized.named_parameters(
      ('below_minimum', [('P1', 50), ('P2', 600)], 'below its minimum'),
      ('over_pool', [('P1', 300), ('P2', 800)], 'more than the funding pool'),
      ('unknown_id', [('P1', 0), ('P2', 600), ('P3', 0)], 'Unknown'),
      ('duplicate_id', [('P1', 0), ('P1', 100), ('P2', 0)], 'more than once'),
      ('missing_id', [('P1', 100)], 'missing'),
      ('negative', [('P1', -100), ('P2', 600)], 'negative'),
      ('fraction', [('P1', 100.5), ('P2', 600)], 'whole-dollar'),
  )
  def test_parse_slate_rejects(self, allocations, message):
    with self.assertRaisesRegex(ValueError, message):
      funding.parse_slate(_slate(allocations), PROPOSALS, 1000)

  def test_parse_slate_rejects_bad_json(self):
    with self.assertRaisesRegex(ValueError, 'not valid JSON'):
      funding.parse_slate('Not JSON', PROPOSALS, 1000)

  def test_render_slate(self):
    slate = funding.parse_slate(
        _slate([('P1', 100), ('P2', 600)], 'Fund both.'), PROPOSALS, 1000)
    self.assertEqual(
        funding.render_slate(slate, PROPOSALS, 1000),
        'Fund both.\n\n'
        '- P1 (Beach cleanup): $100. Reason P1.\n'
        '- P2 (Hampi trip): $600. Reason P2.\n\n'
        'Total: $700 of $1,000 ($300 unallocated)',
    )

  def test_default_question_uses_pool(self):
    self.assertIn('$2,500', funding.default_question(2500))

  @parameterized.named_parameters(
      ('no_proposals', [], 1000),
      ('bad_pool', PROPOSALS, 0),
      ('duplicate_ids', [PROPOSALS[0], PROPOSALS[0]], 1000),
      ('ideal_below_minimum',
       [funding.Proposal('P1', 'T', 'D', 500, 400)], 1000),
  )
  def test_validate_proposals_rejects(self, proposals, funding_pool):
    with self.assertRaises(ValueError):
      funding.validate_proposals(proposals, funding_pool)


if __name__ == '__main__':
  absltest.main()
