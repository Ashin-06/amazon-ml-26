# Dependencies and model provenance

The original pipeline source is supplied under the included MIT license. The fitted baseline model is generated locally from the supplied challenge labels; no pretrained model or externally acquired business record is used. Keep the training input hashes and model manifest with the artifact.

Dependencies are installed from their distributions, not vendored into the archive. Preserve their own distribution license notices. The runtime pins NumPy, SciPy, scikit-learn, LightGBM and RapidFuzz; pytest is a development dependency. Python and SQLite are provided by the selected runtime. Installation of a library does not license unrelated pretrained models.

LightGBM's upstream project uses MIT licensing. Its CPU boosted-tree model is far below the challenge's eight-billion-parameter ceiling. The baseline uses 29 input features and the tree count/leaf settings in its saved configuration. The model manifest records exactly what was fitted; do not substitute a different model without rechecking the competition's restrictions.

Optional pretrained encoders mentioned in the research guide are not downloaded, bundled or required by this baseline. Their licenses/revisions would need separate records if introduced.

The fast pipeline also uses anyascii, DuckDB, pandas, PyArrow and XGBoost 3.0.0. These are installed distributions, not vendored source. Preserve upstream notices with any separately distributed binaries. The fast model has 50 pair and 17 context features; the earlier 29-feature count applies only to SQLite baseline models.
