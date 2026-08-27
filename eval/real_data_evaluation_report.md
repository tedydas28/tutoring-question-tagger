# Evaluation Report: Keyword Baseline on REAL Data

Evaluated on 12 hand-transcribed real questions (spanning all four uploaded files: Algebra, Advanced Math, Geometry and Trigonometry, Problem-Solving and Data Analysis).

## Skill tagging

- Macro F1: 0.542
- Macro Precision: 0.607
- Macro Recall: 0.750

| Skill | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| area_volume | 0.0 | 0.0 | 0.0 | 0 |
| equation_rearrangement | 1.0 | 1.0 | 1.0 | 1 |
| equivalent_expressions | 1.0 | 1.0 | 1.0 | 1 |
| function_notation | 0.0 | 0.0 | 0.0 | 0 |
| linear_eq_one_var | 0.0 | 0.0 | 0.0 | 0 |
| linear_eq_two_var | 1.0 | 1.0 | 1.0 | 1 |
| linear_functions | 0.0 | 0.0 | 0.0 | 1 |
| linear_inequalities | 1.0 | 1.0 | 1.0 | 1 |
| lines_angles | 0.0 | 0.0 | 0.0 | 1 |
| nonlinear_model_interpretation | 1.0 | 1.0 | 1.0 | 1 |
| nonlinear_system_solutions | 1.0 | 1.0 | 1.0 | 1 |
| percentages | 0.0 | 0.0 | 0.0 | 0 |
| ratios_rates | 1.0 | 1.0 | 1.0 | 1 |
| sampling_inference | 1.0 | 1.0 | 1.0 | 1 |
| special_right_triangles | 0.0 | 0.0 | 0.0 | 1 |
| systems_linear_eq | 0.5 | 1.0 | 0.67 | 1 |

### Errors

- **3f5a3602**: gold=['systems_linear_eq'], predicted=['linear_eq_one_var', 'systems_linear_eq']
- **3d1070c9**: gold=['linear_functions'], predicted=['function_notation']
- **002dba45**: gold=['linear_eq_two_var'], predicted=['linear_eq_two_var', 'linear_functions']
- **beca03de**: gold=['nonlinear_model_interpretation'], predicted=['nonlinear_model_interpretation', 'area_volume']
- **a5663025**: gold=['nonlinear_system_solutions'], predicted=['linear_eq_one_var', 'systems_linear_eq', 'nonlinear_system_solutions']
- **6d99b141**: gold=['lines_angles'], predicted=[]
- **2c3aefc9**: gold=['special_right_triangles'], predicted=[]
- **85939da5**: gold=['sampling_inference'], predicted=['percentages', 'sampling_inference']

## Format tagging

- Macro F1: 0.267

### Errors

- **3f5a3602**: gold=['graph_based', 'straightforward'], predicted=['graph_based']
- **f224df07**: gold=['word_problem', 'multi_step'], predicted=['straightforward']
- **beca03de**: gold=['word_problem', 'conceptual'], predicted=['straightforward']
- **a5663025**: gold=['graph_based', 'conceptual'], predicted=['word_problem', 'graph_based']
- **6d99b141**: gold=['diagram_based', 'multi_step'], predicted=['straightforward']
- **2c3aefc9**: gold=['diagram_based', 'multi_step'], predicted=['straightforward']
- **85939da5**: gold=['table_based', 'word_problem', 'conceptual'], predicted=['straightforward']
- **3c8fdc40**: gold=['word_problem', 'straightforward'], predicted=['word_problem']