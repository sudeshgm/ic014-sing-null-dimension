| id | claim | source | AS_SING | AS_fair |
|---|---|---|---|---|
| C1 | DinoViT has the best IS/AS trade-off (highest ratio) of the 5 models | Fig. 3b / Sec. 4.2 / abstract | not testable yet | not testable yet |
| C2 | ResNeXt101 has the lowest IS/AS ratio of the 5 models | Fig. 3b caption | not testable yet | not testable yet |
| C3 | ResNet50 leaks more class semantics into its null space than DinoViT | abstract / Fig. 1 / Fig. 4 | not testable yet | not testable yet |
| C4 | DinoViT has the least class-relevant leakage among the examined networks | Introduction | not testable yet | not testable yet |
| C5 | ResNeXt101 shows high AS (top-2 of the 5) | Sec. 4.2 | not testable yet | not testable yet |
| C6 | ResNet50 has larger and more variable per-class AS than DinoViT | Fig. 4 / Sec. 4.3 | not testable yet | not testable yet |
| C7 | DinoViT per-class |AS| is typically < 1 | Sec. 4.3 | not testable yet | not testable yet |
| C8 | ResNet50 classes Porcupine and Sports-Car leak substantially more than other classes (>= 90th percentile) | Fig. 4 caption | reverses | reverses |
| C9 | No significant correlation between ResNet50 and DinoViT per-class AS orderings | Sec. 4.3 | not testable yet | not testable yet |
| C10 | DinoViT: Jellyfish has substantially larger AS than Arabian Camel (proxy: class-text AS, 10 imgs/class; SING used 30 concepts, ~1300 imgs) | Fig. 5 / Sec. 4.3 | not testable yet | not testable yet |
| C11 | Table 1: DinoViT least steerable; ResNet50/EfficientNet/ResNeXt101 susceptible (proxy: null-removal AS, not IS=40 deg text-gradient steps) | Table 1 / Sec. 4.4 | not testable yet | not testable yet |
| C12 | Fig. 6b / supp. Fig. 12b 13-model IS/AS ordering (holds = Kendall tau > 0 and p < 0.05) | 13-model IS/AS bar plot (digitized in exp0 as Fig. 6b; supp. Fig. 12b) | not testable yet | not testable yet |
| C13 | EVA02 as target space preserves the 5-model ordering | Sec. 4.2 / supp. Fig. 12a | not testable (no EVA02-space translators released) | not testable (no EVA02-space translators released) |
| C14 | Translator robustness: Pearson 0.972 between principal and translated-principal head accuracy | Sec. 4.2 / supp. Fig. 12c | not a comparative leakage claim; not tested | not a comparative leakage claim; not tested |
