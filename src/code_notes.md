# Code notes

Places where binning vs sampling shows up.

## `Spaceline::resample_if_needed`

* // Sampling & Averaging 2 --

**Seems to be missing sampling path**

## `Spacelines::last`

Find the last line for a png, when sampling should throw away line if y_stride doesn't divide,
but may be wrong.

## `SpaceByTime::to_png_and_packed_data`

* // Sample & Averaging 5 --

When `if local_per_x_sample == PowerOfTwo::ONE
                    || self.pixel_policy == PixelPolicy::Sampling`
then takes a single x value rather than averaging over a slice.

## `Spaceline::new`

* // Sampling & Averaging 4 --

## `Spacelines::push_internal`

* // Sampling & Averaging 3

## `Spaceline::compress_if_needed`

* // Sampling & Averaging 1--
