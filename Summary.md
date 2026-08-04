The problem

AI agents that call tools and take actions can be attacked in ways that don't show up in the words they use — a compromised or manipulated agent can skip a required step, call a tool it's never supposed to call, or get talked into auto-approving something it shouldn't. Detecting this by analyzing language (the traditional approach — an LLM judge reading the conversation) misses attacks that are purely structural: nothing said is suspicious, but what happened is wrong. Conversely, some attacks are purely linguistic (social engineering, fake urgency, fake authority) and leave the execution trace looking perfectly normal — a structural check alone misses those.

The naive fix — combine a structural signal and a semantic signal by averaging them — turns out to actively make things worse: averaging dilutes a confident signal whenever the other one is silent, so a real attack that only one detector can see gets voted down by the other detector's silence.

The mechanism

Check structure first, cheaply and deterministically. Only pay for semantic analysis when structure can't already decide.

Two concrete instantiations of that idea, for two different kinds of structural signal:

Rule-based (veto.py): you declare the legal shape of a workflow — which agents may hand off to which, which tools are recognized. Any deviation is a hard, deterministic violation, blocked instantly, no threshold, no model call.
Score-based (fusion.py): when structural risk comes from a trained model rather than a rule, several combination strategies were tested against real data rather than assumed — averaging, max, noisy-or, and confidence-threshold cascading. Only max/noisy-or (and cascade, with caveats) avoid the dilution failure; averaging was proven, empirically, to sometimes have zero real-world recall despite looking fine on a ranking metric like AUC.
What was built

cordon-detect — a small, tested Python library with four working parts:

Detect (veto.py) — deterministic structural rule-checking
Route (fusion.py, routing.py) — pluggable fusion strategies for continuous scores, chosen by evidence rather than a hardcoded default
Evaluate (evaluate.py, the cordon-evaluate CLI) — lets anyone check which fusion strategy actually wins on their own labeled data, reporting both AUC and real threshold-recall, because the two can disagree
Semantic layer (research/) — two working options: a compositional keyword scorer (negation-aware, no dependencies, 98.7% recall / 0% false positives on real generated test data) and a real LLM-backed check (llm_semantic_check.py, 100% recall including cases the keyword scorer structurally can't reach)
How it was validated

Every non-trivial claim was tested against real generated data — not asserted, not tuned to look good:

A synthetic-score illustration was built first, found to be circular (it set the target result and generated data to match it), and replaced with a generator that produces real, varied traces run through cordon's actual code, with results that emerged rather than were preset.
That honest test surfaced the averaging failure mode for real (zero recall on an attack category, hidden behind a fine-looking AUC number).
The keyword scorer went through three rounds of bug-fix-retest (a false positive, then two rounds of fixing a negation-detection bug that kept leaking in new ways) before reaching 0% FPR / 98.7% recall.
The LLM path found a real prompt-injection-adjacent flaw (partially trusting a fabricated risk claim embedded in its own input), a first fix that overcorrected into new false positives, and a second fix confirmed reproducible across three independent runs before being trusted.

Net result: a small library where every design decision — which fusion strategy, whether the semantic layer is reliable, whether a specific bug is actually fixed — is backed by a test that was run, not a claim that was made.