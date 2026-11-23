# Metrics

Scores are **micro-averaged** over the selected split. Counts are summed across images, and each ratio is computed once at the end. A pixel is a crack pixel when its value is greater than zero. Predictions are resized with nearest neighbors to the ground-truth resolution before scoring.

Empty prediction and empty ground truth on a count (no positive pixels anywhere in the split) score as 1. A positive set paired with an empty set scores as 0 for the overlap measures that require agreement.

## Region overlap

True positives, false positives, and false negatives are written \(TP\), \(FP\), and \(FN\).

\[
\mathrm{IoU} = \frac{TP}{TP + FP + FN}, \quad
\mathrm{Dice} = \frac{2\,TP}{2\,TP + FP + FN}
\]

\[
\mathrm{Precision} = \frac{TP}{TP + FP}, \quad
\mathrm{Recall} = \frac{TP}{TP + FN}
\]

## Tolerance F1

A square window of radius \(r\) (Chebyshev distance) absorbs boundary shifts of a few pixels. With \(D_r\) the dilation by that window:

\[
\mathrm{Precision}_r = \frac{|P \cap D_r(G)|}{|P|}, \quad
\mathrm{Recall}_r = \frac{|G \cap D_r(P)|}{|G|}
\]

\[
\mathrm{Tolerance\text{-}F1}_r = \frac{2\,\mathrm{Precision}_r\,\mathrm{Recall}_r}{\mathrm{Precision}_r + \mathrm{Recall}_r}
\]

The default radius is \(r = 2\) on the native 1024×1024 mask. The report field is `tolerance_f1`.

## clDice

Centerline Dice follows Shit et al., CVPR 2021. \(S(\cdot)\) is the iterative soft skeleton, run for 5 iterations by default. Topology precision and sensitivity are accumulated over the split, then combined:

\[
T_{\mathrm{prec}} = \frac{|S(P) \cap G|}{|S(P)|}, \quad
T_{\mathrm{sens}} = \frac{|S(G) \cap P|}{|S(G)|}
\]

\[
\mathrm{clDice} = \frac{2\,T_{\mathrm{prec}}\,T_{\mathrm{sens}}}{T_{\mathrm{prec}} + T_{\mathrm{sens}}}
\]

The report field is `cldice`.
