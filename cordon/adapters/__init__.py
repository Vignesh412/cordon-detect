"""
Adapters convert a third-party trace/span format into cordon's plain
trace format (a list of step dicts — see tokenizer.tokenize()). They
are opt-in: importing cordon itself never pulls in an adapter, so
using cordon doesn't require any tracing SDK as a dependency.

    from cordon.adapters.otel import from_otel_spans
"""
