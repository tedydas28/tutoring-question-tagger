# Evaluation Report: Keyword Baseline Classifier

Evaluated on 20 hand-labeled sample questions.

## Skill tagging performance

- Macro F1: 0.528
- Macro Precision: 0.833
- Macro Recall: 0.583

| Skill | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angle_elevation_depression | 1.0 | 1.0 | 1.0 | 1 |
| area_volume | 1.0 | 1.0 | 1.0 | 2 |
| circles | 1.0 | 1.0 | 1.0 | 1 |
| exponential | 0.0 | 0.0 | 0.0 | 1 |
| function_notation | 1.0 | 1.0 | 1.0 | 1 |
| linear_eq_one_var | 1.0 | 1.0 | 1.0 | 1 |
| linear_eq_two_var | 0.0 | 0.0 | 0.0 | 1 |
| linear_functions | 1.0 | 1.0 | 1.0 | 1 |
| linear_inequalities | 0.0 | 0.0 | 0.0 | 1 |
| percentages | 1.0 | 1.0 | 1.0 | 1 |
| probability | 1.0 | 1.0 | 1.0 | 1 |
| quadratic_factoring | 0.0 | 0.0 | 0.0 | 0 |
| quadratic_formula | 0.0 | 0.0 | 0.0 | 1 |
| ratios_rates | 0.0 | 0.0 | 0.0 | 0 |
| right_triangle_trig | 0.0 | 0.0 | 0.0 | 3 |
| similar_triangles | 1.0 | 0.33 | 0.5 | 3 |
| special_right_triangles | 0.0 | 0.0 | 0.0 | 3 |
| two_var_data | 1.0 | 1.0 | 1.0 | 1 |

### Skill tagging errors

- **q002**: "A flagpole casts a shadow 20 feet long. At the same time, a 5-foot-tall person s..."
  - gold: ['similar_triangles'], predicted: []
- **q003**: "In right triangle PQR, angle Q is 90 degrees, PQ = 6, and QR = 8. What is the le..."
  - gold: ['special_right_triangles'], predicted: []
- **q004**: "A surveyor stands 50 feet from the base of a building and measures the angle of ..."
  - gold: ['angle_elevation_depression', 'right_triangle_trig'], predicted: ['angle_elevation_depression']
- **q006**: "A store sells notebooks for $2 each and pens for $1 each. If Maria spends exactl..."
  - gold: ['linear_eq_two_var'], predicted: []
- **q008**: "Which of the following quadratic equations has no real solutions?"
  - gold: ['quadratic_formula'], predicted: ['quadratic_factoring']
- **q009**: "A population of bacteria doubles every 3 hours. If the initial population is 200..."
  - gold: ['exponential'], predicted: []
- **q013**: "Two triangles have the same shape but different sizes, with corresponding sides ..."
  - gold: ['similar_triangles'], predicted: ['ratios_rates']
- **q014**: "In triangle XYZ, angle X = 45 degrees and angle Y = 45 degrees. If the leg lengt..."
  - gold: ['special_right_triangles'], predicted: []
- **q015**: "If sin(θ) = 3/5 and θ is an acute angle, what is cos(θ)?"
  - gold: ['right_triangle_trig'], predicted: []
- **q016**: "A car rental company charges a flat fee of $30 plus $0.25 per mile driven. Which..."
  - gold: ['linear_inequalities'], predicted: []
- **q020**: "A ladder leans against a wall, forming a 60 degree angle with the ground. If the..."
  - gold: ['right_triangle_trig', 'special_right_triangles'], predicted: []

## Format tagging performance

- Macro F1: 0.464
- Macro Precision: 0.677
- Macro Recall: 0.504

| Format | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| conceptual | 1.0 | 0.5 | 0.67 | 2 |
| diagram_based | 0.0 | 0.0 | 0.0 | 4 |
| graph_based | 0.5 | 1.0 | 0.67 | 1 |
| multi_step | 0.0 | 0.0 | 0.0 | 8 |
| straightforward | 0.56 | 0.9 | 0.69 | 10 |
| table_based | 1.0 | 1.0 | 1.0 | 1 |
| word_problem | 1.0 | 0.12 | 0.22 | 8 |

### Format tagging errors

- **q001**: "Triangle ABC is similar to triangle DEF. If AB = 6, BC = 9, and DE = 4, what is ..."
  - gold: ['straightforward', 'diagram_based'], predicted: ['straightforward']
- **q002**: "A flagpole casts a shadow 20 feet long. At the same time, a 5-foot-tall person s..."
  - gold: ['word_problem', 'multi_step'], predicted: ['straightforward']
- **q003**: "In right triangle PQR, angle Q is 90 degrees, PQ = 6, and QR = 8. What is the le..."
  - gold: ['straightforward', 'diagram_based'], predicted: ['straightforward']
- **q004**: "A surveyor stands 50 feet from the base of a building and measures the angle of ..."
  - gold: ['word_problem', 'multi_step'], predicted: ['straightforward']
- **q006**: "A store sells notebooks for $2 each and pens for $1 each. If Maria spends exactl..."
  - gold: ['word_problem', 'multi_step'], predicted: ['word_problem']
- **q007**: "The graph shown below represents a linear function f. What is the slope of the l..."
  - gold: ['graph_based', 'straightforward'], predicted: ['graph_based', 'diagram_based']
- **q008**: "Which of the following quadratic equations has no real solutions?"
  - gold: ['conceptual'], predicted: ['straightforward']
- **q009**: "A population of bacteria doubles every 3 hours. If the initial population is 200..."
  - gold: ['word_problem', 'multi_step'], predicted: ['straightforward']
- **q010**: "The table below shows the number of hours studied and the corresponding test sco..."
  - gold: ['table_based', 'multi_step'], predicted: ['table_based']
- **q013**: "Two triangles have the same shape but different sizes, with corresponding sides ..."
  - gold: ['word_problem', 'multi_step'], predicted: ['straightforward']
- **q014**: "In triangle XYZ, angle X = 45 degrees and angle Y = 45 degrees. If the leg lengt..."
  - gold: ['diagram_based', 'straightforward'], predicted: ['straightforward']
- **q016**: "A car rental company charges a flat fee of $30 plus $0.25 per mile driven. Which..."
  - gold: ['word_problem', 'multi_step'], predicted: ['straightforward']
- **q018**: "A jar contains 4 red marbles and 6 blue marbles. If one marble is drawn at rando..."
  - gold: ['word_problem', 'straightforward'], predicted: ['straightforward']
- **q019**: "Which of the following best describes the relationship between the graphs of f(x..."
  - gold: ['conceptual'], predicted: ['graph_based', 'conceptual']
- **q020**: "A ladder leans against a wall, forming a 60 degree angle with the ground. If the..."
  - gold: ['word_problem', 'multi_step', 'diagram_based'], predicted: ['straightforward']

## Reading this report

The baseline is deliberately naive (literal keyword matching). Low recall on paraphrased or word-problem-wrapped questions is the expected failure mode - it's the reason to add an LLM-based classifier and/or semantic search, not a sign something is broken. Once real gold-labeled data exists, run `compare_models()` with both the baseline and an `LLMClassifier` instance to see whether the added cost of an API call is actually earning better recall.