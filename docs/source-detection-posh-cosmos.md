# posh/cosmos ACDC source-detection runtime

Reference: `/Users/pshuang/Softwares/Source-detection`, branch `posh/cosmos`,
commit `5b83c44`.

The runtime ensembles two independently trained YOLO11n ONNX detectors from
the reference project's `ONNX` path, running both against the same viewport
ROI and merging their proposals:

- model: `/models/acdc_point_galaxy.onnx`
  - classes: `point`, `galaxy`
- model: `/models/roboflow_universe.onnx`
  - classes: `point`, `arc`, `jet`, `diffuse`, `extended_ring`, `nebula`,
    `shell`, `planet`, `galaxy`
  - trained on a broader 9-class astronomy dataset; included to catch
    point-like sources (e.g. saturated, diffraction-spiked stars) that the
    ACDC model misses or scores too low
- WebGPU disabled
- blob fallback disabled

`galaxy`, `nebula`, `shell`, `diffuse`, `extended_ring`, and `arc` are treated
as "extended" classes and share the wider connected-component Gaussian moment
fit; all other classes (`point`, `jet`, `planet`) use the tighter point-like
fit.

## Viewport inference

1. Extract the currently visible scientific-pixel ROI.
2. Resample it to 640×640 with nearest-neighbour sampling.
3. Exclude CARTA blank pixels (`-FLT_MAX`), calculate `1.4826 × MAD` as the
   lower floor, and use the 99.5 percentile as a robust reference scale.
4. Build two grayscale input tensors for each model: the original linear
   stretch clipped at the 99.5 percentile, plus an `asinh` soft stretch that
   normalizes the actual finite maximum to 1 and preserves gradients inside
   bright, resolved sources. This retains the original model confidence while
   allowing the soft-stretched pass to recover sources hidden by saturation.
   Copy each grayscale plane into every model input channel.
5. Decode either `[1,N,6]` or `[1,4+classes,anchors]` YOLO output, using each
   model's own class list (no cross-model class-id wraparound).
6. Keep model proposals with raw confidence ≥ `0.0001`. Interpret the minimum
   box size as 10 model-input pixels and convert it to the current ROI's image
   scale, so the minimum does not become oversized at high zoom. Retain at
   most the 1000 highest-confidence proposals, then run NMS.
7. NMS operates only within the same class and uses IoU 0.45. A child with raw
   confidence ≥ 0.90 instead uses IoU 0.65. High-confidence proposals have
   sorting priority.
8. Run both input stretches through both ONNX models sequentially at every
   zoom level and concatenate their post-processed detections before the
   shared suppression pipeline below.
   The obsolete connected-component/extended detector is not run.

## Post-processing and display

1. Preserve raw model confidence and map it for display with
   `min(0.99, 1 - exp(-raw × 50))`.
2. Fit intensity-weighted Gaussian moments from the original ROI only to
   calculate integrated and peak flux.
   Point proposals use only pixels connected to the brightest peak inside the
   proposal, preventing a single point ellipse from absorbing nearby sources.
3. Match the reference ACDC `source: "onnx"` display path by replacing the
   raw proposal ellipse with fitted Gaussian moments: point-like regions use
   1σ radii and extended regions use 3σ radii.
4. Apply the raw-confidence ≥5% rule before cross-class spatial filtering.
   Hide a point proposal only when its centre is inside a qualified galaxy
   ellipse whose fitted area is at least four times the point area. This
   prevents a low-confidence or similarly sized galaxy classification from
   erasing the point and then disappearing itself.
5. Do not apply a viewport-relative peak
   flux percentage floor, because it can make valid sources disappear after
   a small zoom or pan changes the brightest source in view.
6. Compute average flux using the reference formula
   `Gaussian total flux / (π radiusX radiusY)` and require at least 0.01% of
   the highest average flux among the remaining candidates.
7. Label ellipses with their detected class name (e.g. `point`, `galaxy`,
   `nebula`) rather than a numeric sequence.
8. Remove repeated regions using the reference project's raw-confidence and
   region-size priority followed by ellipse-centre containment. Extended-class
   pairs also use the reference 0.35 IoU consolidation threshold. Apply
   suppression only within the same detected class name, so results from
   different models still merge (both models emit `point`/`galaxy`) while a
   galaxy cannot erase resolved point sources.
9. Do not apply extended-region representative selection.
10. Track confirmed detections by image-pixel position for the current
    frame/channel/stokes. If a later zoom has no new detection within 3 image
    pixels, retain the historical ellipse while its centre remains in the
    visible ROI. A matching new result replaces the historical result, and a
    channel or Stokes change resets the tracking context.
