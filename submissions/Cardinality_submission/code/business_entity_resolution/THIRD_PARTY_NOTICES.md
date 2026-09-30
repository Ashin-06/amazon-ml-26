# Model and dependency provenance

The neural score caches were produced by fine-tunes of intfloat/multilingual-e5-small and intfloat/multilingual-e5-base. Both upstream model cards declare MIT licensing. The classifiers were fine-tuned on supplied challenge training pairs. The original fine-tuned weights are **not included** in this size-limited archive; their hashes remain in `models/MODELS_MANIFEST.json` for provenance. The included test-score caches run offline and are hashed in `models/COMPACT_MANIFEST.json`. The original models' tensor parameter counts are 117,654,145 and 278,044,417, respectively, below eight billion per model.

- [Small model card](https://huggingface.co/intfloat/multilingual-e5-small)
- [Base model card](https://huggingface.co/intfloat/multilingual-e5-base)
- [Upstream E5 project](https://github.com/microsoft/unilm/tree/master/e5)
- [Upstream MIT license](https://raw.githubusercontent.com/microsoft/unilm/master/LICENSE), retained in `licenses/UNILM_MIT_LICENSE.txt`.

The tree, correction and rescue classifiers were fitted on the supplied challenge data. The package does not include external business identity, geocoding or registry data. Library distributions are installed using the pinned requirements; their own notices and licenses continue to apply. Original pipeline source licensing is preserved in `LICENSE`.
