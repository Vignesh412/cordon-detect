# LangGraph payment-approval demo

This demonstration runs a purchase-approval graph and places Cordon immediately
before the irreversible payment action. It is deterministic, needs no model API
key, and covers three executions:

1. A clean workflow is allowed and payment executes.
2. An agent skips the policy and risk checks; Cordon blocks on structure.
3. Every node runs, but a tool observation contains manipulative override
   language; Cordon escalates to the semantic callback and blocks.

## Run it

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install cordon-detect -r examples/langgraph_demo/requirements.txt
python examples/langgraph_demo/demo.py
```

During local development, use the checkout instead of the PyPI package:

```bash
python -m pip install --upgrade pip
python -m pip install -e . -r examples/langgraph_demo/requirements.txt
python examples/langgraph_demo/demo.py
```

Run one scenario or print the full trace:

```bash
python examples/langgraph_demo/demo.py structural
python examples/langgraph_demo/demo.py semantic --show-trace
```

The semantic callback is intentionally a small deterministic rule so everyone
sees the same result. In a real deployment, replace it with an LLM judge,
classifier, or policy engine. Cordon's job is to enforce structural rules first
and decide whether semantic inspection is needed.
