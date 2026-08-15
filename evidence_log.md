# Bounded range-sweep evidence

## Scope and interpretation

This log records a finite computation performed on 2026-08-15. It is evidence
only for the inclusive integer range `1..1,000,000,000`; it is **not** a proof
of the Collatz conjecture and makes no claim about larger integers. The checker
uses ordinary (unaccelerated) Collatz steps and stops rather than overflowing
its `uint64_t` representation.

## Build

```text
$ cc --version | head -1
cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0

$ cc -std=c11 -O3 -Wall -Wextra -Werror collatz_range_checked.c -o /tmp/collatz_range_checked

$ sha256sum collatz_range_checked.c
17f625bea65aaabd65a2a9039346b1fcb52a2d529e8378598567e90fd7f9e654  collatz_range_checked.c
```

## Complete bounded sweep

The command iterates from the first bound through the second bound,
inclusively. Exit status was `0`; the wall-clock measurement was 465.993
seconds in this environment.

```text
$ /tmp/collatz_range_checked 1 1000000000 | tee data/collatz_1_1000000000.json
{"status":"verified_bounded_range","start":1,"end":1000000000,"checked":1000000000,"maximum_steps":986,"complete":true,"claim":"every input in the inclusive range reached 1 using uint64 arithmetic"}

$ sha256sum data/collatz_1_1000000000.json
e3c2f34719b24faaa01fcfd4ec0d07c0d8385ea3b616ef7655bfa841399cf7c3  data/collatz_1_1000000000.json
```

The equality `checked == end - start + 1 == 1,000,000,000`, together with the
successful status and `complete:true`, is the checker's explicit completeness
statement for this bounded run. The output file itself is ignored under the
repository's generated-data policy; its complete content and digest are
preserved above.
