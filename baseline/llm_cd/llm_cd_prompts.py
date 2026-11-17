def _format_variable_descriptions(feature_names, feature_descriptions, indices):
    lines = []
    for idx in indices:
        name = feature_names[idx]
        desc = feature_descriptions.get(name, name)
        lines.append(f"  {name}: {desc}")
    return "\n".join(lines)


def conditional_independence_prompt(x_idx, y_idx, cond_set, feature_names, feature_descriptions):
    involved = [x_idx, y_idx] + list(cond_set)
    cond_names = [feature_names[idx] for idx in cond_set]
    return f"""If you are an expert in statistics and causal inference, now, please do your best to complete the following task.

# Variable Description
{_format_variable_descriptions(feature_names, feature_descriptions, involved)}

# Task: Conditional Independence Judgment

Are Variable {feature_names[x_idx]} and Variable {feature_names[y_idx]} independent given the condition {cond_names}?

- Score the probability that the stated conditional independence holds.
- This is a closed system, only considering the variables listed above.
- Use your domain knowledge, but remember that correlation does not imply causality.
- If the condition is empty, judge marginal independence between the two variables.

# About Output

Do not output your reasoning process.
The score must be one value from [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0].
A higher value means conditional independence is more likely to hold.

Your final output should follow this exact template:
Answer and Confidence: ___, ___
"""


def directed_edge_prompt(parent_idx, child_idx, feature_names, feature_descriptions):
    indices = [parent_idx, child_idx]
    parent = feature_names[parent_idx]
    child = feature_names[child_idx]
    return f"""You are an expert in causal inference. Your task is to refine a data-driven causal edge.

# Variable Description
{_format_variable_descriptions(feature_names, feature_descriptions, indices)}

# Task

Assume there is a pre-existing causal relationship [{parent}, {child}], where {parent} causes {child}.
Please adjust and refine this edge based on your knowledge and experience.

Use the following options:
- KEEP: "{parent} causes {child}" is correct and should be kept.
- FLIP: reverse the direction because it is more plausible that {child} causes {parent}.
- REMOVE: remove this relationship because no direct causal relationship exists between {parent} and {child}.

# About Output

Do not output your reasoning process.
Output the probability corresponding to each option in JSON format, followed by confidence.
Maintain the exact option names KEEP, FLIP, REMOVE.

Your final output should follow this exact template:
Answer and Confidence: {{"KEEP": _, "FLIP": _, "REMOVE": _}}, ___
"""


def undirected_edge_prompt(a_idx, b_idx, feature_names, feature_descriptions):
    indices = [a_idx, b_idx]
    a_name = feature_names[a_idx]
    b_name = feature_names[b_idx]
    return f"""You are an expert in causal inference. Your task is to orient a data-driven undirected edge.

# Variable Description
{_format_variable_descriptions(feature_names, feature_descriptions, indices)}

# Task: Redirect the Undirected Edge

Please redirect the undirected edge between {a_name} and {b_name}.

- If {a_name} causes {b_name}, output value 1.
- If {b_name} causes {a_name}, output value 0.
- Correlation between two variables does not imply causality.

# About Output

Do not output your reasoning process.

Your final output should follow this exact template:
Answer and Confidence: ___, ___
"""


def cycle_break_prompt(cycle_indices, feature_names, feature_descriptions):
    names = [feature_names[idx] for idx in cycle_indices]
    return f"""You are an expert in causal inference. Your task is to remove a cycle from a causal graph.

# Variable Description
{_format_variable_descriptions(feature_names, feature_descriptions, cycle_indices)}

# Task

These variables form a directed cycle:
{names}

Please either remove or reverse one edge to break the cycle while keeping the causal relationships reasonable.

# About Output

Do not output your reasoning process.
Use one of the following templates:
Remove: <source> and <target>
Reverse: <source> and <target>
"""
