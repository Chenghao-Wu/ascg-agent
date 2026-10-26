---
name: ascg-targets
description: Check the target distributions of a mapped reference before fitting: RDF normalization and range, bond / angle / dihedral histograms, noise and bin widths, state coverage.
---
# Targets

`ascg targets review --study S --task K` covers every target the fit will use: RDFs, bonds, angles and the AA
dihedral distributions (with their unsampled-bin fraction and mean CBT prefactor). For each it reports:
- the bins;
- the statistical noise `rel_se` (block standard error relative to the value in the peak);
- for bonds and angles, the peak resolution `bins_fwhm`;
- range problems.

What to do with its output:
- **warnings: statistically noisy** (block SE above 5 % in the peak): the reference is under-sampled for that histogram.
  Tell the user; longer AA sampling or wider bins (a new `map-aa` run with `--bond-bin-nm` / `--angle-bin-deg`) are
  the remedies, both their decision. The fit smooths over a bin anyway, so moderate noise does not block it.
- **notes: under-resolved** (a smooth peak only a few bins wide, typical of stiff bonds and angles of small rigid
  molecules): informational. The fit can match the histogram at that resolution; finer bins at mapping would resolve
  the shape. Mention it; do not hold the fit for it.
- RDF range beyond half the box: larger system;
- dihedral types with many unsampled bins: the table is extrapolated there; tell the user (usually harmless for a
  stiff backbone);
- missing targets for a state: the reference is incomplete (preflight also fails).
Intermolecular RDFs are the fitted pair targets; chain size is not a target, it is checked in validation.
