# Standalone package checks

Executed locally on 2026-10-24. Exact counts, versions, source hashes and scope are in [package-checks.json](package-checks.json).

- Full package + explicit MyoSuite renderer adapter: **214 passed, 1 skipped**. The skip requires Windows symbolic-link privilege.
- Built-wheel minimal environment: **23 passed, 3 skipped**, with Torch, MyoSuite and USD absent. The three skips need Torch.
- Actual S1 inference extraction parity: eight poses, 95 individual surfaces plus skin, all coordinate differences exactly zero against the original audited source.
- Actual new-package exports: three elbow samples and five full-body S1 anatomy samples, then Blender 4.5.3 scene construction. The full-body scene checks all 96 surfaces.
- Blender comparison: all 96 surfaces preserve cached world coordinates, with no armature/double deformation or duplicate native tendon objects.
- Source adapter checks/apply run against the exact public MyoSuite commit; tests also cover conflicts, repeated application, wrong revisions and invalid/uncommitted checkouts.
- Allocation guards reject advertised oversized array shapes before payload loading. Raw/streaming archives remain numeric-only.
- Wheel and source distribution build successfully; distribution metadata checks pass. Licenses, the adapter patch, CLI and optional-dependency boundaries are retained in built artifacts.

Historical all-five scientific results and their remaining limitations are in [REPORT.md](REPORT.md). Hosted execution is recorded by [GitHub Actions](https://github.com/M-Colley/myosuite-soma/actions), separately from this local snapshot.
