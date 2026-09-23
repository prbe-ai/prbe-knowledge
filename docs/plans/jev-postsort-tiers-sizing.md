# Tier penalty sizing (arm C) on the A/B labels (claude-opus-5-5)

## Volume: what kinds show up

| kind | tier | in engine top-10 (400 traces) | in live candidate pools (chunks, 115 traces) | judged useful (Opus) |
|---|---|---|---|---|
| transcript | 3 | 1521 | 3876 | 74% (1520) |
| file | 2 | 617 | 2113 | 71% (617) |
| run | 0 | 606 | 1034 | 80% (606) |
| gh_commit | 2 | 361 | 2246 | 49% (361) |
| experiment | 0 | 311 | 141 | 89% (311) |
| project | 0 | 223 | 889 | 84% (223) |
| gh_pr | 1 | 103 | 436 | 75% (103) |
| digest | 3 | 88 | 0 | 75% (88) |
| paper | 0 | 79 | 281 | 44% (79) |
| code | 2 | 27 | 404 | 56% (27) |
| team_note | 0 | 16 | 10 | 94% (16) |
| group | 0 | 10 | 15 | 90% (10) |
| gh_release | 1 | 2 | 2 | 50% (2) |
| gh_review | 1 | 1 | 2 | 0% (1) |

## Arms on the same labels (paired per trace; C = tier penalties applied to Jev's probability)

| arm | NDCG@10 | P@4 | P@5 | P@8 | vs B: NDCG diff, 95% CI, wins/losses/ties | traces where C ≠ B |
|---|---|---|---|---|---|---|
| A (today) | 0.934 | 0.776 | 0.761 | 0.743 | -0.020 [-0.031, -0.009] 58/142/164 | 400/400 |
| B (engine order) | 0.954 | 0.815 | 0.797 | 0.755 | +0.000 [+0.000, +0.000] 0/0/364 | 0/400 |
| C 0/0.02/0.05/0.08 | 0.955 | 0.811 | 0.794 | 0.755 | +0.001 [-0.002, +0.005] 47/57/260 | 350/400 |
| C 0/0.03/0.08/0.12 | 0.952 | 0.810 | 0.791 | 0.756 | -0.001 [-0.006, +0.003] 54/70/240 | 371/400 |
| C 0/0.05/0.1/0.2 | 0.951 | 0.800 | 0.786 | 0.754 | -0.003 [-0.008, +0.002] 65/87/212 | 386/400 |

## Rank-1 kind under each arm

| arm | transcript | file | run | gh_commit | experiment | project | gh_pr | digest |
|---|---|---|---|---|---|---|---|---|
| A (today) | 8 | 35 | 55 | 34 | 145 | 76 | 16 | 3 |
| B (engine order) | 140 | 21 | 28 | 14 | 123 | 44 | 7 | 0 |
| C 0/0.02/0.05/0.08 | 78 | 21 | 41 | 11 | 136 | 65 | 24 | 0 |
| C 0/0.03/0.08/0.12 | 65 | 18 | 44 | 10 | 138 | 72 | 28 | 0 |
| C 0/0.05/0.1/0.2 | 41 | 18 | 52 | 14 | 142 | 79 | 29 | 0 |

Jev probabilities found for 400 of 400 traces; a doc without one keeps its engine spacing.
