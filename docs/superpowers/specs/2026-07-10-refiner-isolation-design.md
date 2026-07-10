# RACP baseline refiner isolation design

## Goal

Only the explicitly selected `selective-context` method may construct and run a
refiner. Canonical Naive RAG and Zero-shot runs must never inherit a refiner from
the shared RACP configuration.

## Design

Use two layers of configuration isolation:

1. Set the shared `racp/config.yaml` default `refiner_name` to `null`. This makes
   no-refiner behavior the default for ordinary methods.
2. Set `refiner_name` to `None` explicitly in the method-level configuration
   dictionaries built by `naive()` and `zero_shot()`. This protects paper
   baselines if the shared default changes later.

The existing `selective-context` configuration builder remains responsible for
setting `refiner_name: selective-context`; its behavior and model path are not
changed. IterRetGen and EFC-RAG behavior is outside the change scope.

## Configuration flow

For Naive and Zero-shot, method-level configuration is merged over
`racp/config.yaml`. The resulting `refiner_name` must be `None`, so
`SequentialPipeline` skips `get_refiner()` and does not load GPT-2. For the
`selective-context` method, its explicit method-level value wins and the refiner
continues to load normally.

## Failure handling

No silent fallback is added. Regression tests will fail if a baseline method
again resolves to a non-null refiner or if `selective-context` stops enabling
its refiner.

## Testing

Add focused tests that verify:

- the shared RACP configuration defaults to no refiner;
- `naive()` passes an explicit `refiner_name=None` override;
- `zero_shot()` passes an explicit `refiner_name=None` override;
- the `selective-context` method configuration still enables its refiner;
- the corrected configuration prevents `SequentialPipeline` from calling
  `get_refiner()`;
- existing EFC, QD parser, and title-selection tests still pass.

The tests must avoid loading real retriever, generator, or refiner models.
Formal full-dataset experiments are not part of this change.
