# Retrieval follow-up, before reading new route labels

Hypothesis: first-word corruption excludes true links from the existing rescue route. Probe same-country, same leading house number and same nonempty second core-name word, with different first words. Exclude every existing v24 candidate. Keep the existing raw name/address sum prefilter >=1.5. Bound the raw join at 12 million; one CPU and 700 MB DuckDB memory.

Evaluate tuning references only against the full supplied target pool. Score each tuning reference with its already-frozen out-of-fold v28 rescue model; never refit on these pairs. Fix acceptance cutoff to v28's 0.69. Only add targets not already owned by the base or existing rescue. Keep all tuning references in metric denominators. This is a route feasibility pilot; test deployment would require exporting the added input candidates and checking full-reference competition.

Gate: macro improvement >=.0001, positive lower paired interval, nonnegative country deltas. No gate means no final evaluation or test export. No French rules or identity lookups. Do not infer test recall from this pilot.
