# Sparse feature searches beyond signatures

This directory implements CPU research solvers for finite feature dictionaries. A dictionary provides selected columns, a search for the largest residual correlations, and an explicit transform for comparison. The benchmarks fit independent squared-loss Lasso responses with a free intercept. All three fitting strategies use the same unstandardised features and penalty.

For a residual $r$ and a feature $v$, write $c_v=|\sum_i r_i\phi_v(x_i)|/N$. The tree search retains at most 16 inactive violations, refits on the working set, and repeats until none exceed $\lambda+10^{-7}$. The restricted solver checks stationarity to $8\cdot10^{-8}$, which also prevents an already selected coefficient from occupying a candidate slot. A bound at the current residual prunes for that search. Changing the residual can make a previously pruned branch relevant. The same bounds apply to the negative gradient of a differentiable convex loss with the corresponding scaling. The experiments measure squared loss only.

For example, with this directory on the Python import path, fit a model directly to lists of integer event sequences as follows. Use the same symbol coding for training and prediction. For an alphabet of size eight the valid symbols are 0 through 7.

~~~python
import numpy as np
from solver import fit_tree, predict
from subsequences import Subsequences

training = Subsequences(train_events, alphabet=8, degree=7)
y = np.asarray(train_targets, dtype=np.float64)
model = fit_tree(training, y, alpha=0.01)
test_scores = predict(Subsequences(test_events, alphabet=8, degree=7), [model])[:, 0]
~~~

## Branch bounds

The expressions in this section bound the unnormalised numerator $Nc_v$. Let $r_i^+=\max(r_i,0)$ and $r_i^-=\max(-r_i,0)$. If $L_i\leq\phi_v(x_i)\leq H_i$ for every feature in a branch, a bound is
\[
 \max\left\{\sum_i(r_i^+H_i-r_i^-L_i),\ \sum_i(r_i^-H_i-r_i^+L_i)\right\}.
\]
This follows by separately maximising a signed sum and its negative over the intervals. It retains cancellation whenever the same-sign group has a nonzero lower bound. The implementations include floating-point slack in envelope comparisons and verify their result against an exhaustive fit-time-independent audit.

### Products, rules, Volterra filters, splines and polynomial chaos

Represent an atom $a$ by a vector $f_a$, an input group $g(a)$ and a positive integer cost $d(a)$. A feature chooses at most one atom from each group in increasing group order, with total cost at most $M$. Powers of one variable are alternative atoms in the same group. A categorical literal selects one level, and a linear spline atom selects one hat. A prefix has value $p_i$ on sample $i$.

For strict extensions with remaining budget $b$, define $E_{g,b,i}$ to be the largest absolute product of a nonempty admissible extension over groups $g,g+1,\ldots$. The empty maximum is zero. Splitting by the first group's participation gives
\[
 E_{g,b,i}=\max\!\left\{E_{g+1,b,i},\
 \max_{\substack{g(a)=g\\d(a)\le b}} |f_a(x_i)|\max(1,E_{g+1,b-d(a),i})\right\}.
\]
Thus $|p_i|E_{g,b,i}$ bounds each descendant. For factors of either sign use $\sum_i|r_i p_i|E_{g,b,i}$. When the factors are nonnegative, use the larger of the positive and negative contributions of $r_i p_iE_{g,b,i}$. If all atoms are bounded by one, the suffix maximum of the individual atom moduli is already an envelope at every remaining depth. It drops the budget restriction and costs one reverse pass. A zero prefix removes that sample from every descendant's computation. The explicit comparator stores sparse rules and spline interactions in CSC format. The polynomial experiment uses classical Legendre polynomials on $[-1,1]$. Orthonormal rescaling changes the Lasso penalty in function space and is supported by the cost-aware envelope for factors larger than one.

High-order interaction screening and column generation for rule models predate this work. WHInter combines a working set with branch screening for binary second-order interactions. Sparse Volterra and polynomial chaos estimation give applications to nonlinear system identification and uncertainty quantification. [Nakagawa et al.](https://arxiv.org/abs/1506.08002) [Le Morvan and Vert](https://proceedings.mlr.press/v80/morvan18a.html) [Wei et al.](https://proceedings.mlr.press/v97/wei19a.html) [Kekatos and Giannakis](https://arxiv.org/abs/1103.0769) [Blatman and Sudret](https://doi.org/10.1016/j.jcp.2010.12.021)

### Patterns in text, event streams and graphs

For indicator features let $I(u)$ be the observations containing a prefix or subgraph $u$. Every refinement occurs on a subset of $I(u)$, so all its correlations are bounded by
\[
 B(u)=\max\left\{\sum_{i\in I(u)} r_i^+,\sum_{i\in I(u)}r_i^-\right\}.
\]
Contiguous token patterns use intervals of a suffix array sorted through the maximum pattern length. Repeated occurrences contribute once per document. Arbitrarily gapped patterns use a next-occurrence table. For each surviving document we keep the earliest endpoint of the prefix. Any continuation possible after a later occurrence is also possible after the earliest one. Documents ending there drop out of the strict-descendant bound. This argument applies to unrestricted gaps and binary presence. Counting embeddings or imposing an upper gap limit requires a different state.

The graph dictionary contains connected, vertex-induced motifs with vertex and edge labels. Grow by one adjacent vertex, recording its label and all its edges to earlier vertices. Retain the lexicographically least code over connected orders of the motif. A prefix of a least code must itself be least, since a better ordering of the prefix could be extended to the full set. Project all matching embeddings when extending a canonical node and count each graph once. The permutation canonicaliser is intended for small motifs and caps their order at six. A production-scale graph miner would use a specialised canonical labelling algorithm.

An additional gapped-pattern comparator enumerates every observed pattern but builds only one training column per distinct support. Hash matches are verified by exact equality of the sorted observation indices. If two training columns agree, replacing their coefficients by their sum on one column preserves all fitted values and cannot increase the penalty. Thus this comparator has the same optimal training objective. The representative can behave differently on held-out recordings. Pattern extensions still need enumerating because two equal supports may have different earliest endpoints.

Pattern growth, residual-gradient bounds and safe pruning have direct precedents. The tests here concern the practical benefit of the same working-set strategy on these dictionaries. [Pei et al.](https://www.cs.sfu.ca/~jpei/publications/span.pdf) [Gsponer et al.](https://doi.org/10.1007/978-3-319-71246-8_3) [Nakagawa et al.](https://arxiv.org/abs/1602.04548) [Yoshida et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10724371/)

### Scattering paths

Each feature averages a nonnegative modulus state across time or graph vertices. The time-series implementation uses periodic, undecimated Haar filters. If $H_j=(I+\mathrm{shift}_{2^j})/2$, the $j$th wavelet is $(I-\mathrm{shift}_{2^j})H_{j-1}\cdots H_0/2$. Each wavelet annihilates constants, has operator norm at most one on $\ell_2$, and convolution kernel of $\ell_1$ norm at most one. The modulus preserves $\ell_1$ and $\ell_2$ norms. For a nonnegative prefix state $s_i$ of length $T_i$, every strict descendant's mean lies in $[0,E_i]$ with
\[
 E_i=\min\{\overline{s_i},\ \|s_i-\overline{s_i}\mathbf1\|_2/\sqrt{T_i}\}.
\]
The first term follows by $\ell_1$ contraction at every step. The second follows by removing the mean before the next wavelet, $\ell_2$ contraction throughout the suffix, and Cauchy-Schwarz at pooling. Use the signed split above. The full and pruned Haar transforms share a fused compiled traversal with reusable depth-wise buffers.

For an undirected graph with combinatorial Laplacian $L$, use $P=I-L/(2\max(1,d_{\max}))$. Its spectrum lies in $[0,1]$. The filters $I-P,P-P^2,P^2-P^4,\ldots$ annihilate constants and have $\ell_2$ operator norm at most one. The centred RMS term therefore applies graph by graph, including graphs of different sizes. No $\ell_1$ contraction is needed. Features here are global means. A different pooling, path-dependent normalisation or resampling operator requires its own bound. [Mallat](https://arxiv.org/abs/1101.2286) [Gama et al.](https://arxiv.org/abs/1806.08829) [Ioannidis et al.](https://arxiv.org/abs/2001.09882)

### Landmarks and shapelets

Let $W_i$ be the fixed-length windows of sample $i$, or a singleton for ordinary landmark features. Define $d_i(z)=\min_{w\in W_i}\|w-z\|_2$ and $\phi_z(x_i)=\exp(-\gamma d_i(z)^2)$. The minimum distance to a fixed set is 1-Lipschitz. In a template ball of centre $p$ and radius $R$ we have
\[
 \exp[-\gamma(d_i(p)+R)^2]\ \leq\ \phi_z(x_i)\
 \leq\ \exp[-\gamma\max(0,d_i(p)-R)^2].
\]
Insert these endpoints into the signed interval bound. For the usual closest-window distance itself, the endpoints are $\max(0,d_i(p)-R)$ and $d_i(p)+R$. The search bisects along the coordinate of largest spread, batches exact distances at leaves and caches up to 64 pivot-distance vectors. Windows and templates in both shapelet experiments undergo the same per-window normalisation before applying the Euclidean metric. [Ye and Keogh](https://www.cs.ucr.edu/~eamonn/shaplet.pdf)

### Interval indicators

For one ordered covariate, let $P_j$ be the residual sum in all bins with index below $j$. The feature for $[a,b)$ has signed numerator $P_b-P_a$. Within a group of right endpoints $L\le b\le H$ at fixed $a$, the exact maximum is
\[
 \max\left\{P_a-\min_{L\le j\le H}P_j,\ \max_{L\le j\le H}P_j-P_a\right\}.
\]
All initial groups $b>a$ get their maxima with one reverse pass over the bins. Keep the best $k$ groups and repeatedly extract the best interval. Its remaining right endpoints split into the intervals on either side of the maximiser. Discarding the other initial groups is valid because the retained groups each contain a distinct interval at least as good. Ties may return any $k$ maximisers. The initial work is linear in the bins after histogramming. At most $2k$ restricted endpoint scans follow. A separate matrix-free comparator scores all endpoint pairs from the same histograms. This is a regression use of the classical $k$ maximum segment sums problem. [Cheng et al.](https://www.csie.ntu.edu.tw/~kmchao/papers/2006_TCS_KMAXSUM.pdf) [Optimal subgroup discovery in numerical data](https://pmc.ncbi.nlm.nih.gov/articles/PMC7206336/)

### Boolean Fourier features

Write $\chi_S(x)=(-1)^{\sum_{j\in S}x_j}$. At depth $l$ the inclusion decisions on the first $l$ bits are fixed by $U$. Aggregate the signed residual on each observed remaining bit string $z$ as $h_z=\sum_{i:x_{i,l:}=z}r_i\chi_U(x_i)$. For any remaining subset $T$, its signed numerator is $\sum_z h_z\chi_T(z)$ and its modulus is bounded by $\sum_z|h_z|$. The children fold pairs of entries into $h_{0z}+h_{1z}$ and $h_{0z}-h_{1z}$, respectively. Keeping only observed suffixes avoids a $2^d$ array. With five or fewer undecided bits the code evaluates the block by a Walsh-Hadamard transform. A second comparator performs the full Walsh-Hadamard transform on the residual histogram at every search. This is deterministic empirical correlation search, distinct from learning a Fourier spectrum via queries. [Kushilevitz and Mansour](https://doi.org/10.1137/0222080)

## Problems and evaluation

| Family | Dictionary and protocol |
|---|---|
| Event patterns | GunPoint, ECG200 and Wafer use the official train/test partitions. Average 30 consecutive blocks, difference their means, symbolise slopes by eight training octiles, and admit every binary-presence gapped word through length seven. Stratify 24% of the official training set for penalty selection and refit on the full official training set. |
| Sound | Use the paper's 600/300/2200 Spoken Arabic Digit split. Compare integrated PCA-transformed MFCC coordinates with the underlying 24 cepstral frames recovered by differencing. Per-channel pooled Haar uses four scales and increasing-scale paths through order three. Event patterns use eight training-only minibatch k-means states and unrestricted gaps through length six. Keep the 600-recording fit after validation. |
| Tabular rules | Use positive categorical literals on distinct Mushroom attributes through order three. Use all nonempty one-dimensional intervals on Bike Sharing calendar and weather predictors with at most 48 training bins per variable. Split the hourly data chronologically 60/20/20. The response is rentals divided by 1000 for fitting. Multiply RMSE by 1000 for reporting. Observed weather is available as a predictor. |
| Text and graphs | Use binary word/punctuation n-grams of lengths one to four on UCI SMS Spam. Remove duplicate tokenised documents and conflicting duplicate labels before splitting. On MUTAG use all induced connected labelled motifs through four vertices, or diffusion scattering of one-hot atom labels with three filters through order three. The diffusion uses unweighted adjacency. Use the first of five stratified folds for timings and all five for the graph accuracy comparison. SMS and Mushroom use the same 60/20/20 stratified protocol. Treat a missing Mushroom attribute as a category. |
| System identification and simulation | The Cascaded Tanks fit uses only input lags 0, 1, 2, 3, 5, 7, 10, 14, 19, 25, 32, 40 and 50, scaled using the training input range. Include all monomials through degree three. Score the official test simulation after the first 50 samples. The Ishigami benchmark adds five irrelevant uniform inputs and uses classical Legendre products through total degree six. Friedman 1 uses 12 variables, unit Gaussian noise and tensor products of five local hat functions through interaction order three. |
| Scattering and templates | GunPoint also uses globally pooled Haar on all ordered filter paths through order four at seven scales. Shapelet candidates are all distinct normalised length-16 training windows starting at stride four. Evaluate every sliding window in an observation. Test both raw closest-window distance and its Gaussian similarity. The spatial control uses a 96 by 96 RBF landmark grid and noisy observations of a mixture of four bumps. Simulations use 512/256/1024 train/validation/test rows. |
| Boolean Fourier features | Use all nonconstant Walsh characters of the ten bits encoding the six MONK-2 categorical attributes. Use only the training sample for validation/refitting and the archive's 432-point test universe for scoring. The universe includes the training combinations. The control script also computes accuracy on unseen combinations. |

In every case select a single ratio from 0.30, 0.10, 0.03 and 0.01. Multiply it by each response's own training-only full-dictionary $\lambda_{\max}$. Classification maximises validation accuracy, then minimises validation MSE. Regression minimises validation MSE. Exact ties favour the larger penalty. Freeze the ratio and recompute $\lambda_{\max}$ on the final fitting rows. On the benchmark machine, rebuilding the Arabic paths and labels from the public archive reproduced the saved arrays element for element.

The fixed-split tasks have published comparisons. The graph control evaluates a normalised Weisfeiler-Lehman subtree kernel with a validation-selected SVM on exactly our five folds. The scripts also run Euclidean 1-NN on the three UCR splits, a linear SVM on SMS, a decision tree on Mushroom, gradient boosting on Bike Sharing and a linear finite impulse response model on Cascaded Tanks. [UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) [WL kernels](https://www.jmlr.org/papers/v12/shervashidze11a.html) [Speech data](https://archive.ics.uci.edu/dataset/195/spoken+arabic+digit) [Cascaded Tanks benchmark](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks) [SMS collection](https://doi.org/10.1145/2034691.2034742) [MONK comparison](https://www.cs.cmu.edu/~thrun/papers/thrun.MONK.html)

## Reproducing the measurements

Use Python 3.12 and the supplied [requirements](requirements.txt). From the repository root run the following commands. The scripts fetch and verify public dataset archives. To use an existing paper split, add the option shown below. Using the raw Spoken Arabic archive reconstructs the same arrays on the benchmark environment.

~~~shell
python -m pip install -r benchmarks/implicit_features/requirements.txt
python benchmarks/implicit_features/test_bounds.py -v
python benchmarks/implicit_features/experiment.py --output output/feature_study --data output/feature_data --repeat 3
python benchmarks/implicit_features/penalty_path.py --output output/feature_study --data output/feature_data --repeat 3
python benchmarks/implicit_features/comparisons.py --output output/feature_study --data output/feature_data
python benchmarks/implicit_features/comparisons.py --timed --output output/feature_study --data output/feature_data
python benchmarks/implicit_features/summarize.py --output output/feature_study --plot output/feature_study/comparison.png
~~~

Use `--arabic PATH/TO/prepared.npz` for the two timing scripts, `--cases` to select applications, and a new output directory for independent repetitions. The suite resumes completed jobs and rotates method order. The result files include validation curves, individual timings, selected support sizes, objectives, global KKT violations, held-out predictions and memory measurements. Reference fits, tuning and exhaustive audits are outside the single-fit timings. The separate penalty-path benchmark includes calibration, dictionary construction, four warm fits and four validation predictions. Its full baseline builds the training matrix once.

The principal comparator is Celer on the explicit training matrix. Another comparator runs the identical batched working-set algorithm on the explicit matrix. The interval and Walsh cases also include specialised matrix-free comparators. Peak resident memory covers feature construction and fitting in a fresh process. Report the increase above RSS immediately before fitting, using 2 ms polling and the OS high-water mark if it rises. Absolute peak RSS is available too. Python, JIT warm-up on unrelated tiny data and loaded input arrays form part of the process baseline. Prediction and exhaustive auditing occur after the fit memory observation. Times are single-core wall times with BLAS and OpenMP limited to one thread. [Celer](https://proceedings.mlr.press/v80/massias18a.html)

### Published pattern-mining comparator

PMOpt implements safe pattern pruning and coordinate descent for pattern dictionaries. Compare its squared-loss Lasso with the same binary features, free intercept, maximum degree and penalty on Mushroom, SMS and the three UCR datasets. After mapping the coefficients back into our feature index, audit every training correlation. The supplementary timing includes calculation of the maximum penalty for both solvers. The tree starts with observations in memory. PMOpt starts with their symbolic CSV, and its executable time includes loading, both penalty trials and model output. The cost of converting observations to CSV is reported separately. Its absolute memory is the native child process high-water observed at 1 ms intervals. [Yoshida et al.](https://arxiv.org/abs/2306.13561) [PMOpt source](https://github.com/takeuchi-lab/pmopt)

Build PMOpt from commit `196db0a78af3c244df2808ab9d2c1955de688932`, with Boost headers available to the compiler. From its checkout apply `git apply --ignore-space-change --unidiff-zero /path/to/pmopt.patch`, then run `cmake -S pmopt -B build -DCMAKE_BUILD_TYPE=Release` and `cmake --build build --config Release`. The patch exposes the existing maximum-skip variable as a command-line argument and supplies portability fixes for GCC 13 and Windows. The upstream default is zero gaps. A maximum skip equal to the input length permits arbitrary gaps. The experiment uses the upstream optimiser and screening rule.

~~~shell
python benchmarks/implicit_features/pmopt_compare.py --binary /path/to/pmopt --output output/feature_study --data output/feature_data
python benchmarks/implicit_features/pmopt_compare.py --binary /path/to/pmopt --output output/feature_study --data output/feature_data --methods pmopt-default --repeat 1
~~~

The strict run requests a relative dual gap of `1e-12`. The additional run uses PMOpt's default `1e-4`. Each output records its measured KKT error, including when it exceeds the common accuracy target, and any 180 s or 2 GiB child-process limit. A published predictive comparison on the same UCR partitions is SAX-SEQL. Its symbolic transformation and loss differ from ours. [Le Nguyen, Gsponer and Ifrim, Table IV](https://researchrepository.ucd.ie/rest/bitstreams/25473/retrieve)
