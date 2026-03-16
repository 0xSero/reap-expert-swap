# 30pct isolation comparison

## armA-dynamic-prewarmed

- accuracy: 0.32
- avg request time s: 6.701375
- avg sample time s: 6.873364
- avg swap time s: 0.054684
- avg swap control plane s: 0.054684
- avg cold swap time s: 0.0
- avg warm swap time s: 0.054684
- avg warm change/reuse swap time s: 0.0 / 0.054684
- avg completion tokens/s: 6.35906
- avg warm completion tokens/s: 6.35906
- cold/warm swap count: 0 / 25
- same-signature rows: 25
- zero-copy rows: 25
- nonzero copy rows: 0
- dynamic signature count: 1

## armB-forced-static

- accuracy: 0.32
- avg request time s: 6.277379
- avg sample time s: 6.449026
- avg swap time s: 0.058783
- avg swap control plane s: 0.058783
- avg cold swap time s: 0.065229
- avg warm swap time s: 0.058515
- avg warm change/reuse swap time s: 0.0 / 0.058515
- avg completion tokens/s: 6.506437
- avg warm completion tokens/s: 6.566963
- cold/warm swap count: 1 / 24
- same-signature rows: 24
- zero-copy rows: 25
- nonzero copy rows: 0
- dynamic signature count: 1

## armC-cold-included

- accuracy: 0.04
- avg request time s: 7.558749
- avg sample time s: 1.251707
- avg swap time s: 0.063217
- avg swap control plane s: 0.012643
- avg cold swap time s: 0.064857
- avg warm swap time s: 0.010468
- avg warm change/reuse swap time s: 0.0 / 0.010468
- avg completion tokens/s: 6.720496
- avg warm completion tokens/s: 7.29347
- cold/warm swap count: 1 / 24
- same-signature rows: 24
- zero-copy rows: 25
- nonzero copy rows: 0
- dynamic signature count: 1

