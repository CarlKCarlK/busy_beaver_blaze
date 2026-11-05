# Changelog

## Unreleased

### Visualization

- Added an experimental multiresolution preview with Linear and Square Root time
  scaling, and Linear or fitted 1/Y horizontal stretching.
- Simplified the time selector and removed the pyramid size/debug readout.

- Enabled Speed/Sampling for Square Root time scaling without averaging samples.
- Tightened program/control spacing and synchronized both scale selections with
  the URL hash in both directions.

### Acknowledgments

**Nonlinear time scaling:** Thanks to
[opfromthestart](https://github.com/opfromthestart), whose triangular-number
sampling of Turing-machine executions inspired the square-root time scaling
in Busy Beaver Blaze. Their observation was that machines repeatedly sweeping
between expanding boundaries often grow in width roughly as the square root
of elapsed steps.
