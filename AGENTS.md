# Project Instructions

## Writing And Plotting Standards

- Prefer standard domain conventions, names, and symbols across prose, formulas, code outputs, and plots. For optimization reports, use `L` for loss and `Delta L = L_after - L_before` for loss change unless there is a specific reason to define another quantity.
- Do not introduce custom symbols or alternate names for standard quantities unless they add clarity. If a custom symbol is necessary, define it once, justify it, and use it consistently.
- Keep definitions uniform across README, notebooks, scripts, generated CSV columns, plot labels, and captions. If compatibility requires both signs of the same quantity, explicitly state the mapping, for example `delta_loss = L_before - L_after = -loss_change`.
- Use the same sign convention for related formulas and plots whenever possible. If one figure uses loss and another uses improvement, explain the change explicitly.
- Any number shown in prose, a table, legend, or annotation should have an interpretation. For fitted slopes, coefficients, thresholds, or scales, state what theoretical or empirical quantity they estimate and how to read the sign and magnitude.
- Optimize the report for first-pass readability: each retained formula, table, and plot should answer a concrete experimental question or support interpretation of a result.

- Do not create or save plots that are not directly tied to a stated experimental question, unless they are explicitly marked as temporary debugging artifacts.
- Every saved plot should make the tested dependence clear from its title, axes, legend, and surrounding explanation.
- Before plotting, identify the expected theoretical dependence and choose axes that make that dependence easy to visually compare against theory.
- Prefer transformations that linearize the predicted relationship when they do not obscure the meaning of the axes. If a theory predicts a linear relation, plot it on linear axes with a fitted line rather than using log or normalized axes by default.
- Prefer ordinary linear axes when the expected shape is already visually meaningful, for example a local quadratic loss-change curve versus step size. Do not use log axes if they make the shape look artificially sharp or harder to verify.
- For saturating theoretical curves with arbitrary fitted scales, prefer normalized axes when they improve readability. For example, plot `epsilon_opt / epsilon_max` versus `B / B_noise` so the curve saturates at `1` and can be visually checked against `x / (1 + x)`.
- If axes are normalized, label them with the exact normalization, for example `epsilon_opt / epsilon_max` or `B / B_noise`.
- Before finalizing a plot, verify that every axis label exactly matches the plotted data, including whether values are raw, normalized, clipped, averaged, or transformed.
- If error bars are shown, identify what they represent in the legend, axis label, caption, or nearby report text, for example standard error across sampled minibatches.
- Optimize every plot for first-glance readability: the expected pattern should be visible without reverse-engineering the code or mentally undoing unnecessary axis transforms.
- Avoid large blank plot regions when they do not carry meaning. Tighten axis limits to the data and theory region unless preserving a boundary such as zero or an asymptote is necessary for interpreting the prediction.
- Keep related plots visually and semantically consistent when possible: use the same sign convention, axis meaning, labels, and color/marker semantics across figures.
- When a plot has characteristic regimes or regions, label the region that matters for interpretation rather than over-emphasizing an isolated boundary point. For example, in a step-size loss curve, label the too-large-step region as loss-increasing instead of only marking the zero-change crossing.
- Avoid plotting quantities merely because they are available. A plot should answer a concrete question or diagnose a concrete failure mode.
- If a plot is retained despite being exploratory, document what question it answers and what pattern should be expected.
