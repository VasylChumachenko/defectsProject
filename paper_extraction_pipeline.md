# Optical Descriptor Extraction Pipeline

> **Формули** подано у LaTeX-сумісному форматі. Рекомендації щодо вставки у Word Online — в кінці документа.

## Data sources

Diffuse reflectance spectroscopy (DRS) data in the g-C₃N₄ literature are most commonly reported either as Kubelka–Munk-transformed spectra or as Tauc plots. In this work, both representations were used as valid point-data sources for downstream analysis, depending on what was available in the original article.

The Kubelka–Munk (K–M) function provides a standard transformation of diffuse reflectance data into a quantity proportional to absorption for optically thick powdered samples and is commonly written as

$$F(R_\infty) = \frac{(1 - R_\infty)^2}{2\,R_\infty}\,,\qquad\text{(1)}$$

where $R_\infty$ is the diffuse reflectance of the sample relative to a non-absorbing reference. In the present study, digitized K–M spectra were treated as the primary source for deriving optical descriptors whenever available.

The Tauc plot is routinely used to estimate the optical band gap ($E_\mathrm{g}$) by fitting the linear region of the absorption edge and extrapolating the fit to the energy axis. For g-C₃N₄, the indirect-transition form, $(\alpha h\nu)^{1/2}$ vs. $h\nu$, is most commonly used; therefore, this representation was preferentially used in our workflow. When only a direct-transition Tauc plot, $(\alpha h\nu)^2$ vs. $h\nu$, was reported, it was converted to the indirect form prior to further processing. Likewise, when only absorbance-like spectra (or K–M spectra) were available, the indirect Tauc representation was reconstructed from the digitized data using the same conversion procedure.

This unified preprocessing step allowed all samples to be mapped into a consistent optical-descriptor space for subsequent feature extraction and spectral classification.

## Baseline correction

When the same article contained a reference sample with a literature-reported band-gap value, a baseline correction was applied to compensate for vertical offsets in digitized absorbance data (e.g., due to detector offset, parasitic scattering, or figure-axis truncation). A constant $\delta^*$ was subtracted from the raw absorbance:

$$\alpha_{\mathrm{corr}}(E) = \alpha_{\mathrm{raw}}(E) - \delta^*\,,\qquad\text{(2)}$$

where $\delta^*$ was determined by a grid search ($n = 100$ linearly spaced values up to 30% of the peak absorbance, followed by a local refinement step of 40 points) that minimized the absolute error between the extracted band gap and the known literature value for the calibrator sample, $|\,E_\mathrm{g}^{\,\mathrm{fit}}(\delta) - E_\mathrm{g}^{\,\mathrm{lit}}\,|$. The same $\delta^*$ was then applied to all other samples from the same article. Negative absorbance values after subtraction were clamped to a small positive floor ($10^{-12}$) to maintain numerical stability.

## Band gap determination

The linear region of each Tauc plot was identified using a sliding-window linear regression procedure. An initial window covering 25% of the available data points (minimum 5 points) was scanned along the energy axis. At each window position, ordinary least-squares linear regression was applied (`scipy.stats.linregress`), and the coefficient of determination ($R^2$) was recorded.

Only candidate windows with a positive slope ($a > 0.1$) and a preliminary band-gap estimate of $E_\mathrm{g} \ge 1.8$ eV were retained. The preliminary band gap was calculated as the energy-axis intercept of the fitted line, $E_\mathrm{g} = -b/a$, where $a$ and $b$ are the slope and intercept, respectively.

The highest-$R^2$ candidate window was then iteratively expanded in both directions, one point at a time, while preserving all of the following criteria: $R^2 \ge 0.997$, positive slope, and a physically valid band-gap estimate. To avoid extending the fit into adjacent non-linear spectral regions, the expansion was limited to 40% of the total number of data points. After expansion, 5% of the points (minimum 2 points) were trimmed from each end of the window to reduce edge-related artifacts, and the final $E_\mathrm{g}$ value was obtained from a new linear fit over the trimmed interval.

The lower bound of 1.8 eV was imposed to exclude unphysical fits and is consistent with the reported electronic structure range of g-C₃N₄.

A confidence label was assigned to each $E_\mathrm{g}$ estimate using two criteria: the final $R^2$ value and the relative size of the fitted interval. Estimates were labeled *high confidence* for $R^2 \ge 0.997$ with at least 10% of points retained, *medium confidence* for $R^2 \ge 0.990$ with at least 5% of points retained, and *low confidence* otherwise.

If the best-ranked candidate window yielded $E_\mathrm{g} < 1.8$ eV after fitting, the algorithm proceeded to the next-ranked window until a physically valid solution was obtained.

## Urbach energy determination

The Urbach energy ($E_\mathrm{u}$), which characterizes the width of the exponential absorption tail and is commonly used as a descriptor of structural/electronic disorder, was extracted from the sub-band-edge region of each spectrum. The analysis follows the Urbach relation,

$$\alpha(E) = \alpha_0 \exp\!\left(\frac{E}{E_\mathrm{u}}\right),\qquad\text{(3)}$$

which describes an exponential increase in absorption near the absorption edge. In logarithmic form, $\ln\alpha = aE + b$, the Urbach energy is obtained as $E_\mathrm{u} = 1/a$, where $a$ is the slope of the fitted linear region.

The search interval for the Urbach tail was defined adaptively with respect to the previously estimated band gap as $[E_\mathrm{g} - 0.35,\; E_\mathrm{g} - 0.08]$ eV, and the fitted window width was additionally capped at 0.28 eV. This tighter anchoring (compared to a wider search) was used to consistently target the near-edge exponential region while excluding both the interband transition regime (close to $E_\mathrm{g}$) and the noisy deep sub-gap baseline.

Prior to logarithmic transformation, a baseline correction was applied by subtracting the median absorbance in the low-energy region below the Urbach search zone ($E < E_\mathrm{g} - 0.55$ eV), to compensate for detector offset and parasitic scattering/background contributions. The baseline-corrected absorbance was then smoothed using a Savitzky–Golay filter (window length = 11 points, polynomial order = 2) before calculating $\ln\alpha$, since smoothing after the logarithmic transform can amplify noise in low-absorption regions.

The optimal linear fitting region in $\ln\alpha$ vs $E$ was identified by an exhaustive pairwise search over all valid start–end index combinations $(i,\,j)$, subject to the following constraints: (i) minimum window width $\ge 0.12$ eV; (ii) maximum window width $\le 0.28$ eV; (iii) minimum number of points $\ge\max(5,\;20\%\text{ of available points})$; (iv) positive slope; (v) $E_\mathrm{u}$ within a physically plausible range (15–800 meV); and (vi) $R^2 \ge 0.97$. Each candidate window was scored using a composite criterion,

$$S = R^2 - 0.15\,\frac{\sigma_a}{|a|}\,,\qquad\text{(4)}$$

where $\sigma_a/|a|$ is the relative standard error of the slope from ordinary least-squares regression. This score balances goodness-of-fit and slope robustness, penalizing windows that yield high $R^2$ but unstable slope estimates.

To improve robustness, an ensemble strategy was applied: all candidate windows scoring within 0.005 of the best score were collected, and the final $E_\mathrm{u}$ was taken as the median value across this top-$K$ set. This ensemble approach reduces the sensitivity of $E_\mathrm{u}$ to the exact choice of fitting boundaries.

A confidence label was assigned to each $E_\mathrm{u}$ estimate based on three criteria: the fit quality of the representative window ($R^2$), the relative slope error ($\sigma_a/|a|$), and the ensemble spread (IQR of the top-$K$ $E_\mathrm{u}$ values relative to the median). An estimate was labeled *high confidence* for $R^2 \ge 0.995$, relative slope error < 10%, and relative ensemble spread < 10%; *medium confidence* for $R^2 \ge 0.98$, relative slope error < 20%, and relative ensemble spread < 25%; and *low confidence* otherwise.

## Sub-gap absorption index

To quantify defect-related absorption below the Urbach tail, a sub-gap absorption index ($A_\mathrm{sub}$) was introduced. This descriptor captures optical absorption in excess of the fitted Urbach contribution and is interpreted as a proxy for localized sub-gap states associated with structural defects (e.g., carbon/nitrogen vacancies, cyano groups, or oxygen-containing surface functionalities). Although $A_\mathrm{sub}$ is primarily interpreted in terms of defect-related sub-gap states, it may also include contributions from residual baseline imperfections, surface complexes, or digitization noise in low-intensity regions; therefore, it is used here as a comparative descriptor rather than a direct state-density measure.

The absorbance spectrum was first normalized by the mean absorption in the UV region $[3.2,\;3.6]$ eV, where g-C₃N₄ typically exhibits strong interband absorption, while remaining sufficiently far from the band-edge fitting region:

$$\tilde{\alpha}(E) = \frac{\alpha(E)}{\langle\alpha\rangle_{3.2\text{–}3.6\;\mathrm{eV}}}\,.\qquad\text{(5)}$$

This normalization makes $A_\mathrm{sub}$ dimensionless and improves comparability across samples with different signal scales (e.g., due to optical path length, concentration, or figure scaling in digitized spectra).

The integration window was defined relative to the fitted Urbach region, with the upper bound set to the lower-energy edge of the Urbach fit ($E_\mathrm{Urbach,end}$) rather than to the extrapolated band gap. This choice ensures that only absorption below the exponential Urbach tail is quantified. The lower bound was set to $E_\mathrm{Urbach,end} - 0.5$ eV, with a hard lower-energy limit of 1.3 eV imposed by the available spectral range. To account for incomplete low-energy coverage in some digitized spectra, a coverage metric was defined as the ratio of the actual integration width to the ideal 0.5 eV window.

Within the selected window, the Urbach contribution was extrapolated as

$$\tilde{\alpha}_\mathrm{Urbach}(E) = \frac{\exp(aE + b)}{\langle\alpha\rangle_\mathrm{norm}}\,,\qquad\text{(6)}$$

where $a$ and $b$ are the slope and intercept obtained from the Urbach fit. The excess sub-gap absorption was then computed as

$$\Delta\tilde{\alpha}(E) = \max\!\bigl(0,\;\tilde{\alpha}(E) - \tilde{\alpha}_\mathrm{Urbach}(E)\bigr),\qquad\text{(7)}$$

so that only positive deviations from the Urbach baseline were retained. The raw sub-gap absorption was calculated by trapezoidal integration,

$$A_\mathrm{sub}^\mathrm{raw} = \int \Delta\tilde{\alpha}(E)\,\mathrm{d}E\,.\qquad\text{(8)}$$

To ensure comparability across spectra with different usable energy ranges, the final index was normalized by the actual integration width $\Delta E_\mathrm{actual}$:

$$A_\mathrm{sub} = \frac{A_\mathrm{sub}^\mathrm{raw}}{\Delta E_\mathrm{actual}}\,,\qquad\text{(9)}$$

which represents the mean excess sub-gap absorption per electronvolt.

A confidence label was assigned based on spectral coverage and point density in the integration window: *high confidence* for coverage $\ge 80\%$ and $\ge 10$ points; *medium* for coverage $\ge 50\%$ and $\ge 5$ points; *low* for coverage $\ge 30\%$ and $\ge 3$ points; otherwise *invalid*. Samples with coverage below 50% were excluded from subsequent clustering analysis to reduce artifacts caused by truncated spectral data.

## Derived spectral descriptors

In addition to the three primary optical quantities ($E_\mathrm{g}$, $E_\mathrm{u}$, $A_\mathrm{sub}$), four secondary descriptors were computed from the fitted spectral parameters to provide a richer characterization of the optical response.

**Disorder ratio.** A dimensionless disorder parameter was defined as

$$r_\mathrm{dis} = \frac{E_\mathrm{u}}{E_\mathrm{g}}\,,\qquad\text{(10)}$$

where both quantities are expressed in electronvolts. This ratio normalizes the tail width by the band gap, allowing comparison of relative disorder across samples with different $E_\mathrm{g}$ values. A higher $r_\mathrm{dis}$ indicates a proportionally broader tail and greater electronic disorder relative to the fundamental gap.

**Sub-gap slope.** The energy dependence of absorption in the sub-gap region (i.e., within the $A_\mathrm{sub}$ integration window) was quantified by fitting a linear model in logarithmic absorbance space:

$$\frac{\mathrm{d}(\ln\alpha)}{\mathrm{d}E}\bigg|_\mathrm{sub\text{-}gap}\,.\qquad\text{(11)}$$

A steeper (more positive) slope indicates that sub-gap absorption follows a shallow exponential characteristic of tail states, whereas a flatter slope suggests deeper, more localized mid-gap levels. The fit required at least 5 valid data points with positive absorbance.

**Absorption-edge asymmetry.** To characterize the shape of the absorption onset, the asymmetry of $\ln\alpha$ around $E_\mathrm{g}$ was quantified as

$$\eta = \frac{\bigl|\,\mathrm{d}(\ln\alpha)/\mathrm{d}E\,\bigr|_{E > E_\mathrm{g}}}{\bigl|\,\mathrm{d}(\ln\alpha)/\mathrm{d}E\,\bigr|_{E < E_\mathrm{g}}}\,,\qquad\text{(12)}$$

where the slopes above and below $E_\mathrm{g}$ were obtained from linear fits in symmetric energy windows whose half-width was set to $0.75\times\Delta E_\mathrm{trans}$ (clamped to $[0.12,\;0.40]$ eV), $\Delta E_\mathrm{trans}$ being the width of the Tauc-plot linear region. Each side of the fit required at least 4 valid points; otherwise, $\eta$ was marked as unavailable. Values of $\eta \approx 1$ indicate a symmetric edge; $\eta > 1$ indicates a sharper onset above $E_\mathrm{g}$ relative to the sub-edge tail.

**Urbach fit residual.** The quality of the single-exponential Urbach model was assessed by computing the root-mean-square (RMS) deviation between the data and the fit within the Urbach region:

$$\varepsilon_\mathrm{Urb} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\bigl(\ln\alpha_i - (aE_i + b)\bigr)^2}\,,\qquad\text{(13)}$$

expressed in $\ln\alpha$ units. Elevated $\varepsilon_\mathrm{Urb}$ values indicate that the absorption tail cannot be described by a single exponential, suggesting the presence of multiple disorder mechanisms or distinct defect contributions.

---

## Рекомендації щодо формул у Microsoft Word Online

Word Online має обмежений редактор формул порівняно з десктопною версією. Рекомендовані підходи:

### Варіант 1: Десктопна версія Word (найкраще)

1. Відкрити документ у Word (десктоп), вибрати **Insert → Equation** (або `Alt+=`).
2. В полі рівняння ввести LaTeX-синтаксис напряму — Word конвертує його автоматично. Наприклад:
   - `F(R_\infty) = \frac{(1-R_\infty)^2}{2R_\infty}` → відобразиться як формула.
   - `E_u = 1/a` → підрядковий індекс автоматично.
3. Зберегти і повернутися до Word Online — формули залишаться.

### Варіант 2: Word Online (обмежений)

1. **Insert → Equation** — доступні лише шаблони.
2. Для простих формул (індекси, дроби) — використовувати шаблони з меню.
3. Для складних — набрати у десктопній версії або Google Docs, потім скопіювати.

### Варіант 3: Google Docs → Word

1. Набрати формули в Google Docs через **Insert → Equation** (має зручний візуальний редактор).
2. Експортувати як `.docx` — формули зберігаються.

### Варіант 4: MathType / Wiris

Плагін [MathType for Word Online](https://www.wiris.com/en/mathtype/) працює через Office Add-ins і підтримує LaTeX-введення повністю. **Insert → Get Add-ins → MathType**.

### Рекомендація

Для статті найнадійніше — **варіант 1** (десктопний Word): набрати формули через LaTeX-синтаксис у Equation Editor, зберегти — далі редагувати текст можна і в Word Online. LaTeX-код кожної формули наведено вище між `$$...$$`.

---

# Bootstrap Perturbation Analysis and Parameter Stability

The stability of the extracted spectral parameters ($E_\mathrm{g}$, $E_\mathrm{u}$, $A_\mathrm{sub}$, and the four derived descriptors $r_\mathrm{dis}$, sub-gap slope, edge asymmetry $\eta$, Urbach residual $\varepsilon_\mathrm{Urb}$; nine features in total) with respect to experimental noise and preprocessing choices was evaluated using a bootstrap-based perturbation analysis. For each of the 209 spectra in the dataset, 100 perturbed realizations were generated under three independent perturbation modes.

**(1) Noise injection.**
Heteroscedastic Gaussian noise was added to the absorbance spectrum according to:

$$\alpha'(E_i) = \max\!\bigl(0,\;\alpha(E_i) + \varepsilon_i\bigr), \qquad \varepsilon_i \sim \mathcal{N}\!\bigl(0,\;\sigma_\mathrm{noise}\cdot s_i\bigr),\qquad\text{(14)}$$

where $s_i = \max\!\bigl(|\alpha(E_i)|,\;P_5(|\alpha|)\bigr)$ is the local noise scale, clipped from below at the 5th percentile of the absolute signal to prevent vanishing noise in near-zero regions, and $\sigma_\mathrm{noise} = 0.02$ (i.e., 2% of the local signal magnitude). This formulation reflects the heteroscedastic character of practical spectrophotometric uncertainty, where stronger signals typically exhibit larger absolute fluctuations. The lower bound prevents degenerate perturbation in baseline-dominated spectral regions.

**(2) Point dropout.**
A random 15% subset of spectral points was removed from each spectrum to simulate reduced effective resolution, incomplete digitization, or isolated point loss/outliers. A minimum of 30 points was retained in each perturbed spectrum to ensure stable execution of the fitting routines.

**(3) Smoothing-window variation.**
The Savitzky–Golay smoothing window length was randomly sampled from {5, 7, 9, 11, 13} points (default: 7 points for the Tauc stage), while keeping the polynomial order fixed at 2, in order to assess the sensitivity of the extracted parameters to preprocessing settings. For samples with baseline correction, the same correction offset $\delta^*$ determined from the unperturbed analysis was applied prior to perturbation.

For each perturbed realization, the full extraction pipeline was re-run: Tauc transformation and $E_\mathrm{g}$ estimation $\rightarrow$ Urbach-tail fitting and $E_\mathrm{u}$ estimation $\rightarrow$ sub-gap integration and $A_\mathrm{sub}$ calculation $\rightarrow$ derived descriptor computation ($r_\mathrm{dis}$, sub-gap slope, $\eta$, $\varepsilon_\mathrm{Urb}$). In addition to the three primary descriptors, the Tauc-edge slope and the Tauc linear-region width were also tracked to monitor the stability of the underlying band-gap fit geometry.

The resulting parameter distributions were then compared with the corresponding values obtained from the unperturbed spectra to quantify (i) precision, expressed as the relative error $\mathrm{RE} = \sigma_\mathrm{boot}/|\mu_\mathrm{orig}|$, where $\sigma_\mathrm{boot}$ is the bootstrap standard deviation and $\mu_\mathrm{orig}$ is the original (unperturbed) parameter value; and (ii) systematic bias, expressed as the mean deviation from the baseline estimate.

---

# Feature Preprocessing and Clustering

## Feature selection

From the set of nine optical descriptors computed for each sample, three were retained for the spectral-type clustering analysis: (i) the optical band gap $E_\mathrm{g}$ (eV), which separates polymorphic phases and broadly different electronic structures; and (ii) the sub-gap absorption index $A_\mathrm{sub}$, which captures defect-related absorption below the Urbach tail. Together, these two features were used for the first (macro) clustering stage. A third descriptor, the disorder ratio $r_\mathrm{dis} = E_\mathrm{u}/E_\mathrm{g}$, was used in combination with $A_\mathrm{sub}$ for the second (sub-clustering) stage.

The remaining features—Tauc-edge slope, transition width, raw $E_\mathrm{u}$, edge asymmetry, sub-gap slope, and Urbach residual—were excluded from the clustering input after exploratory analysis showed that they either correlated strongly with the retained descriptors, introduced noise-dominated variance, or degraded cluster separation.

## Feature transformation

Prior to clustering, the feature vectors were transformed using the Yeo–Johnson power transform (`PowerTransformer`, scikit-learn) to reduce skewness, stabilise variance, and improve the suitability of distance-based and likelihood-based clustering methods. This transformation was selected based on inspection of the raw descriptor distributions, which showed substantial right-skewness (particularly for $A_\mathrm{sub}$). The transformation was applied independently at each stage: to $(E_\mathrm{g},\;A_\mathrm{sub})$ for macro-clustering and to $(A_\mathrm{sub},\;r_\mathrm{dis})$ for sub-clustering.

## Stage 1: Macro-clustering using Gaussian Mixture Model

The first stage of clustering (macro-clustering) was performed using a Gaussian Mixture Model (GMM) with full covariance matrices (`GaussianMixture`, scikit-learn), applied to the two-dimensional transformed feature space $(E_\mathrm{g},\;A_\mathrm{sub})$. The number of mixture components was fixed at $K = 2$, reflecting the dominant bimodal structure observed in the data: a low-defect group (cluster A, lower $A_\mathrm{sub}$) and a high-defect group (cluster B, higher $A_\mathrm{sub}$). Ten random initialisations were used to reduce sensitivity to local optima.

Cluster labels were assigned deterministically: the component with the lower mean $A_\mathrm{sub}$ was labeled A and the other B. GMM was chosen at this stage because it provides probabilistic (soft) cluster assignments, which are useful for identifying samples with uncertain membership near cluster boundaries.

## Stage 2: Sub-clustering with Spectral Clustering

To resolve finer structure within the high-defect macro-cluster, a second-stage analysis was performed using Spectral Clustering (`SpectralClustering`, scikit-learn, radial basis function affinity, 10 initialisations) on the transformed feature space $(A_\mathrm{sub},\;r_\mathrm{dis})$.

Sub-clustering was applied only to cluster B; cluster A was retained as a single unsplit group (`--no-split A`), as its internal structure did not exhibit further separable sub-populations in the relevant feature space. Within B, the number of sub-clusters was fixed at $K = 2$, yielding two sub-groups (B.1 and B.2) with distinct combinations of defect absorption level and relative disorder. Sub-cluster labels were assigned deterministically so that B.1 had the lower mean $A_\mathrm{sub}$.

Spectral Clustering was used for the sub-stage instead of GMM because the sub-group boundary in $(A_\mathrm{sub},\;r_\mathrm{dis})$ space is not well-described by ellipsoidal Gaussian contours; the graph-partitioning approach of Spectral Clustering captures the non-convex separation more effectively.

This two-stage procedure—GMM macro-clustering on $(E_\mathrm{g},\;A_\mathrm{sub})$ followed by Spectral Clustering on $(A_\mathrm{sub},\;r_\mathrm{dis})$ within B only—produced a three-group spectral-type hierarchy (A, B.1, B.2).

## Transition vector analysis

To characterise the direction and magnitude of spectral changes induced by defect engineering and doping—rather than absolute spectral properties—modification vectors were constructed for each within-article reference → modified comparison. For each pair, the modification vector was defined as:

$$\boldsymbol{\delta}_i = (\Delta E_\mathrm{g},\;\Delta E_\mathrm{u},\;\Delta A_\mathrm{sub})_i = \bigl(E_\mathrm{g}^\mathrm{mod} - E_\mathrm{g}^\mathrm{ref},\;E_\mathrm{u}^\mathrm{mod} - E_\mathrm{u}^\mathrm{ref},\;A_\mathrm{sub}^\mathrm{mod} - A_\mathrm{sub}^\mathrm{ref}\bigr),\qquad\text{(15)}$$

where "ref" and "mod" denote the reference (control) and modified samples, respectively. In articles containing multiple modified samples sharing a common reference, each modification produced a separate vector. When an article contained multiple reference samples, the reference with the lowest $A_\mathrm{sub}$ (most pristine) was selected as the primary reference for all comparisons within that article. If no reference sample was reported, a virtual reference was constructed from the article-level spectral mean. An additional derived vector component, $\Delta r_\mathrm{dis} = r_\mathrm{dis}^\mathrm{mod} - r_\mathrm{dis}^\mathrm{ref}$, was computed for the sub-cluster-level analysis.

## Transition classification

Rather than clustering the modification vectors in $\Delta$-space, transitions were classified into discrete types using a rule-based scheme operating on the two disorder-related components $\Delta E_\mathrm{u}$ and $\Delta A_\mathrm{sub}$. The classification proceeded as follows:

1. **Noise thresholds.** The $2\sigma$ bootstrap noise levels for each feature (obtained from the perturbation analysis described above) defined a dead zone within which changes were considered indistinguishable from measurement noise.

2. **Normalisation.** The raw deltas were converted to standardised z-scores using the dataset standard deviations, $z_{E_\mathrm{u}} = \Delta E_\mathrm{u}/\mathrm{SD}(\Delta E_\mathrm{u})$ and $z_{A_\mathrm{sub}} = \Delta A_\mathrm{sub}/\mathrm{SD}(\Delta A_\mathrm{sub})$, placing both components on equal footing.

3. **Disorder vector metrics.** The disorder magnitude $D = \sqrt{z_{E_\mathrm{u}}^2 + z_{A_\mathrm{sub}}^2}$, the net disorder $D_\mathrm{net} = (z_{E_\mathrm{u}} + z_{A_\mathrm{sub}})/\sqrt{2}$ (projection onto the equi-disorder axis), and the purity $P = |D_\mathrm{net}|/D$ (alignment with the equi-disorder direction) were computed.

4. **Type assignment.** Transitions were classified as:
   - *Perturbative*: both $|\Delta E_\mathrm{u}|$ and $|\Delta A_\mathrm{sub}|$ below their respective $2\sigma$ noise thresholds;
   - *Disordering*: $D_\mathrm{net} > 0$ (net increase in disorder);
   - *Ordering*: $D_\mathrm{net} < 0$ (net decrease in disorder).

5. **Band-gap shift annotation.** Independently of the disorder classification, each transition was annotated with an $E_\mathrm{g}$ shift label (*narrowing*, *widening*, or *stable*) based on whether $|\Delta E_\mathrm{g}|$ exceeded its $2\sigma$ noise threshold.

The resulting transition types and $E_\mathrm{g}$ shift annotations were cross-tabulated against spectral-type transitions (reference cluster → modified cluster) and synthesis tags to identify systematic correlations between modification strategies and spectral outcomes.

---

# Clustering Stability Validation

The robustness of the three-group spectral-type hierarchy (A, B.1, B.2) was assessed through three complementary bootstrap procedures that mirror the two-stage clustering pipeline described above.

## Stage 1–2: Macro-clustering stability

**Subsample bootstrap.** In each of 500 iterations, 20% of the 209 samples were randomly excluded. A GMM with $K = 2$ and full covariance was re-fit on the Yeo–Johnson-transformed $(E_\mathrm{g},\;A_\mathrm{sub})$ feature space of the retained subset. Because the GMM label assignment is arbitrary up to permutation, re-estimated labels were aligned to the original (full-dataset) labels using the Hungarian algorithm, which maximises the overlap between reference and resampled assignments by solving a linear sum assignment on the confusion matrix.

From the 500 resampled partitions, two stability metrics were computed:

(i) A *consensus matrix* $\mathbf{C}$, where element $C_{ij}$ records the fraction of iterations in which samples $i$ and $j$ (when both present) were assigned to the same cluster:

$$C_{ij} = \frac{\sum_b \mathbb{1}[\hat{y}_i^{(b)} = \hat{y}_j^{(b)}]\;\mathbb{1}[i,j \in S_b]}{\sum_b \mathbb{1}[i,j \in S_b]}\,,\qquad\text{(16)}$$

where $S_b$ is the retained sample set in iteration $b$ and $\hat{y}_i^{(b)}$ is the aligned cluster label of sample $i$.

(ii) *Per-sample stability*, defined as the fraction of iterations (when the sample was present) in which the bootstrap label agreed with the original full-dataset label.

Additionally, at each iteration the BIC-optimal number of components was recorded (scanning $K = 1,\dots,6$) to assess whether the choice of $K = 2$ was supported under resampling. The adjusted Rand index (ARI) between the original and resampled label vectors was computed as a global partition-agreement measure.

**Feature perturbation bootstrap.** To evaluate sensitivity to measurement noise in the feature values (rather than sample composition), Gaussian noise with standard deviation equal to 5% of each feature's empirical standard deviation was added to the full $(E_\mathrm{g},\;A_\mathrm{sub})$ matrix in each of 500 iterations. The GMM($K = 2$) was then re-fit, labels were aligned via the Hungarian algorithm, and ARI and per-sample stability were computed as above.

## Stage 3: Nested (full-hierarchy) stability

To assess the stability of the complete three-group hierarchy—not only the macro-partition—a nested bootstrap was performed. In each of 300 iterations, 20% of samples were excluded, and the full two-stage pipeline was re-executed on the retained subset:

1. GMM($K = 2$) on Yeo–Johnson-transformed $(E_\mathrm{g},\;A_\mathrm{sub})$ → macro-labels (A / B), aligned to the original partition;
2. Spectral Clustering($K = 2$, RBF affinity) on Yeo–Johnson-transformed $(A_\mathrm{sub},\;r_\mathrm{dis})$ within cluster B only → sub-labels (B.1 / B.2), with deterministic ordering by mean $A_\mathrm{sub}$.

Cluster A was retained unsplit, producing full labels {A, B.1, B.2} in each iteration. The full-label ARI was computed between the resampled and original label vectors, and per-sample full-label stability was recorded. The effective number of sub-clusters recovered within B was also tracked across iterations.

## Summary of results

Subsample bootstrap ($n = 500$, drop 20%): mean ARI = 0.818 ± 0.120; BIC selected $K = 2$ in 69.6% of iterations ($K = 3$ in 29.4%, $K = 4$ in 1.0%); per-sample stability: 85.6% of samples above 80%, mean = 0.953. Feature perturbation bootstrap ($n = 500$, $\sigma = 5\%$): mean ARI = 0.865 ± 0.063; per-sample stability mean = 0.965. Nested bootstrap ($n = 300$, drop 20%): macro ARI = 0.821 ± 0.118; full (three-group) ARI = 0.801 ± 0.116; sub-clustering within B recovered $K = 2$ in 100% of iterations; full-label stability mean = 0.926.

The macro-partition was moderately stable overall, with instability concentrated at the boundary of cluster A (mean stability = 0.914, 19/67 samples below 80%) rather than within cluster B (mean stability = 0.971). The sub-clustering within B was highly stable, with $K = 2$ recovered in all iterations. The BIC preference for $K = 3$ in ~30% of subsample runs reflects the tendency to split boundary-region samples into a separate group under reduced sample size, rather than genuine three-cluster structure in the full dataset.

---

# LLM-Assisted Extraction of Synthesis Descriptors

## Overview

To systematically link spectral signatures with synthesis conditions, structured metadata was extracted from the Experimental sections of the source publications using a large language model (GPT-4o, OpenAI; temperature = 0.1). The extraction was performed in a single-pass prompt that requested a JSON response conforming to a predefined schema. The model input comprised: (i) the article title, (ii) the full Experimental text (truncated to 12 000 characters), and (iii) a list of spectral-data file names associated with the article (derived from the digitised DRS dataset). JSON structure was enforced during generation using the `response_format: json_object` constraint.

The LLM was instructed to identify each distinct g-C₃N₄ sample described in the article and to extract (a) free-text synthesis descriptors and (b) a set of standardised categorical tags (defined below). In addition, the LLM mapped each spectral-data file name to the corresponding sample label reported in the article, following explicit naming rules (lowercase, dash-separated conventions). To avoid ambiguous mappings, each file name was permitted to match at most one sample.

## Two-stage synthesis schema

A preliminary analysis revealed that a flat extraction schema conflated backbone-formation conditions with post-treatment conditions—for example, recording a 300 °C reduction-step temperature as the sole `temperature_C` when the g-C₃N₄ backbone had been formed at 600 °C. To resolve this ambiguity, a two-stage schema (designated `v2_staged`) was adopted. For each sample, the LLM was instructed to decompose the synthesis protocol into:

**Stage 1 — Backbone formation**: polycondensation of the nitrogen-rich precursor to form the g-C₃N₄ network (fields: `backbone_temperature_C`, `backbone_atmosphere`, `backbone_method`, `backbone_duration_bin`, `backbone_heating_rate_C_min`). The prompt explicitly required extraction of the final holding (regime) temperature in multi-step heating programmes, not intermediate ramp temperatures. Temperatures reported in Kelvin were converted to °C.

**Stage 2 — Modification / post-treatment** (optional): any subsequent processing of the already-formed g-C₃N₄ (fields: `is_post_processed`, `mod_method`, `mod_temperature_C`, `mod_atmosphere`, `mod_duration_bin`, `mod_agent`, `mod_notes`). A controlled vocabulary of nine modification methods was defined: re-calcination, solvothermal, chemical etching, gas treatment, wet impregnation, ultrasonication, ball milling, plasma treatment, and other.

A detailed decision tree was included in the prompt to distinguish the two stages: `is_post_processed = true` only when a separate second step was applied to already-formed g-C₃N₄ (e.g., re-calcination, exfoliation, gas treatment), whereas samples that differed from the reference solely in their one-step synthesis conditions (different precursor, temperature, atmosphere, or additive) were assigned `is_post_processed = false`.

When backbone conditions were not explicitly described (e.g., commercial g-C₃N₄ or synthesis "according to ref. [X]" without details), the corresponding fields were set to null/unknown rather than inferred.

## Standardised per-sample tags

Eight categorical tags were assigned to each sample — six describing backbone/universal properties and two describing the modification step:

**Precursor family** — primary nitrogen-rich precursor forming the g-C₃N₄ backbone: melamine, urea, thiourea, cyanamide (including dicyandiamide/DCDA), or other. In mixtures, only the backbone-forming precursor was assigned; secondary components were recorded separately as co-precursors.

**Calcination temperature bin** — backbone final holding temperature, discretised into: < 520 °C, 520–559 °C, 560–599 °C, and ≥ 600 °C. This tag was derived exclusively from `backbone_temperature_C`, not from the modification temperature.

**Atmosphere class** — gas environment during backbone thermal treatment: air/ambient, N₂, inert (Ar, He), reducing (H₂-containing), etching/reactive (NH₃), CO₂-generating (e.g., NaHCO₃ decomposition), or unknown. Derived from `backbone_atmosphere`.

**Primary synthesis route** — dominant backbone pathway: direct thermal polycondensation, hydro-/solvothermal pre-treatment, supramolecular pre-assembly, template-assisted, or other/unknown.

**Dopant class** — type of intentional doping: undoped, non-metal (P, S, B, F, O), metal (e.g., Fe, Cu, La), or co-doped/multi-element.

**Morphology form** — declared product form: bulk, nanosheets/ultrathin, porous/holey, tubular, 3D macroporous, or unknown.

**Modification method** — how the post-treatment was performed: none (no post-processing), re-calcination, solvothermal, chemical etching, gas treatment, wet impregnation, ultrasonication, ball milling, plasma treatment, or other.

**Modification atmosphere class** — classified gas/medium during the post-treatment step: none, inert, N₂, air, reducing, etching/reactive, liquid, vacuum, or unknown. Derived from `mod_atmosphere`.

Two additional categorical descriptors were recorded as supplementary fields: backbone synthesis method (thermal polymerisation, solvothermal, supramolecular, other) and backbone duration bin (< 2 h, 2–4 h, 4–8 h, > 8 h). Each sample was classified as *reference* (pristine/unmodified g-C₃N₄ used as the within-article baseline) or *modified* (any sample that differs from the reference — including doped, exfoliated, post-treated, or differently-synthesised variants). At the article level, the UV–Vis/DRS spectrophotometer brand and model were recorded when reported.

## Post-processing and validation

All LLM outputs were subjected to deterministic post-processing: (i) tag values were validated against the allowed vocabulary using case-insensitive matching, with fallback to default values when no match was found; (ii) `calcination_temperature_bin` was recalculated from numeric `backbone_temperature_C` values to enforce consistency between the free-text and categorical representations; (iii) duplicate file-name assignments were resolved by retaining the first valid match; and (iv) for backward compatibility with analysis scripts developed under the flat schema, legacy columns (`temperature_C`, `atmosphere`, `synthesis_method`, `duration_bin`) were computed automatically from the corresponding backbone fields.

## Reproducibility infrastructure

Extraction was managed by a run-orchestration script (`run_extraction.py`) that created a timestamped run directory containing the full prompt configuration snapshot (a copy of the Python config module), resulting CSV/JSON files, and a manifest recording the config version, model identifier, prompt hash (SHA-256 of the full prompt template), git commit hash, and SHA-256 hashes of all output files. Incremental re-extraction was supported via a `--skip-cached` mode that seeded the output from the most recent completed run with the same config version (with a hard-stop safety check preventing cross-version caching). Run comparison (`--diff`) was provided to verify consistency between extraction runs.

The final extraction (config `v2_staged`, GPT-4o) processed 85 articles and returned 349 sample records, of which 243 (69.6%) were successfully matched to spectral-data files in the digitised dataset. A random subset of extracted records was manually inspected to verify tag validity and file–sample mappings.

---

# Dataset Overview

The literature corpus comprised 107 research articles reporting UV–Vis DRS spectra of g-C₃N₄-based photocatalysts. A total of 392 digitised spectra from these articles were submitted to the automated feature-extraction pipeline. Of these, 182 spectra were excluded at the extraction stage: 130 spectra from 36 articles failed entirely (all samples in those articles produced invalid fits), and a further 52 individual spectra from 30 other articles were rejected due to insufficient data points, excessive noise, or failure to identify a physically valid linear region during the band-gap or Urbach-energy determination (i.e., no fit satisfying the $E_\mathrm{g} \ge 1.8$ eV, $R^2$, or slope constraints could be obtained).

The resulting dataset comprised 210 successfully extracted spectra from 71 articles, with 1–8 spectra per article (median 2, mean 3.0), typically including at least one pristine/reference g-C₃N₄ spectrum alongside modified samples. The articles were sourced from studies spanning a range of defect-engineering and compositional-modification strategies, and were catalogued under four broad thematic categories according to the dominant modification motif: nitrogen vacancy defects (41 articles, 124 spectra), element-doped systems (13 articles, 42 spectra), cyano group defects (7 articles, 23 spectra), and carbon vacancy defects (10 articles, 21 spectra). These categories were used for data management only and did not enter the analysis pipeline; all samples were treated uniformly regardless of the declared modification type.

After applying quality-control filtering based on the minimum sub-gap integration coverage ($A_\mathrm{sub}$ coverage $\ge 50\%$), 209 spectra remained for clustering and downstream statistical analyses. Of these, 12 spectra from 7 articles were subjected to calibrator-based baseline correction prior to feature extraction.

The extracted optical descriptors spanned the following ranges: $E_\mathrm{g}$ = 1.88–2.96 eV, $E_\mathrm{u}$ = 39–778 meV, $A_\mathrm{sub}$ = 0.000–0.378, and $r_\mathrm{dis}$ ($E_\mathrm{u}/E_\mathrm{g}$) = 0.016–0.346. Confidence labels for the three primary descriptors were distributed as follows: $E_\mathrm{g}$ — 67 high, 91 medium, 52 low; $E_\mathrm{u}$ — 133 high, 68 medium, 9 low; $A_\mathrm{sub}$ — 186 high, 23 medium, 1 low.

LLM-assisted extraction of synthesis metadata yielded 169 spectra (from 63 articles) with matched synthesis descriptors, including 51 reference and 118 modified samples. From these, 118 within-article reference → modified transition vectors were constructed across 52 articles (16 using virtual references computed from the article-level spectral mean). Of the 118 transitions, 54 corresponded to post-processing modifications and 64 to parallel synthesis variations.
