# MMX
## Abstract

We introduce MMX (Media Mix Modeling with Latent Extensions), a fully Bayesian framework for marketing attribution that reconciles aggregate revenue signals with privacy-constrained fingerprinting data like SKAN. MMX captures latent monetization dynamics, seasonal effects, and channel-level biases, allowing for simultaneous inference of organic and paid revenue components. By accounting for structural halo and cannibalization effects and aligning attribution with inferred monetization state, MMX improves interpretability and robustness in scenarios where last-touch methods fail.

Through over 100 realistic simulations, MMX consistently outperforms SKAN-derived response curves in attribution accuracy, especially for low-spend or bias-prone channels, while maintaining a "do-no-harm" property when identifiability is limited. We extend MMX into a Spend Decision Framework, which estimates the probability of profitability at each spend scale by sampling from the model's posterior. This enables channel-specific media planning based not on point estimates, but on the conditional likelihood of positive return.


## Introduction

The constraints of modern digital advertising—particularly in mobile app ecosystems—have fundamentally altered the landscape of performance measurement. With the depreciation of user-level tracking and the introduction of privacy-preserving attribution systems such as Apple’s SKAN, advertisers are left with severely degraded visibility into the true causal impact of their marketing efforts. These limitations are further compounded by the presence of attribution biases like halo and cannibalization, which are not addressed by naive heuristics or deterministic last-touch assignment.

In this context, traditional attribution models fall short—either by oversimplifying the structure of influence across channels or by relying on granular data that is no longer accessible. In response, we introduce a Bayesian framework that directly models aggregate revenue generation while explicitly accounting for latent revenue dynamics, structured attribution distortion, and variable inter-channel identifiability.

Rather than attempting to reverse-engineer user-level behavior, the model identifies latent signals of paid and organic contributions to revenue by conditioning on observed media inputs and adjusting for seasonal events and long-run trends. Attribution distortion from SKAN data is modeled explicitly through structured bias terms, while inference is carried out jointly to preserve identifiability and maintain internal consistency across modeled pathways.

To validate the approach, we conduct a suite of simulations across a variety of marketing environments, stress-testing the model’s ability to outperform SKAN-derived response curves under different bias regimes and spend distributions. These simulations show that the model achieves its intended goal: doing no harm where SKAN is reliable, and directionally improving performance when attribution distortions are substantial.

In addition to retrospective evaluation, the model directly informs budgeting decisions via a Spend Decision Framework, which estimates the probability of profitability at different spend levels for each channel. This discrete, probabilistic framing enables more robust spend recommendations than traditional point estimates or mean-variance optimizations, especially when operating far from the data’s historical support.

This work aims to offer both a methodologically grounded and practically useful alternative to existing attribution pipelines in a world where user-level ground truth is no longer available—and may never return.


## Limitations of SKAN and Last-Touch Attribution

The core limitation that motivates this work is the structural inadequacy of SKAN (StoreKit Ad Network) and similar last-touch attribution frameworks to capture the true causal influence of marketing activities. While SKAN was introduced by Apple as a privacy-preserving alternative to deterministic, user-level attribution, it introduces a range of measurement issues that fundamentally compromise its utility as a causal inference tool.

### 1. Attribution Bias: Halo and Cannibalization

SKAN relies on a last-touch windowed approach to assigning credit for installs, but installs are often influenced by both paid and organic factors. Paid media can create a halo effect, generating interest that converts later through organic means, such as branded search or app store discovery. Conversely, SKAN may over-attribute conversions to paid media that would have occurred organically—especially in the presence of strong brand equity or timed organic bursts. This is known as cannibalization. Both effects distort the observed relationship between spend and revenue and can create highly misleading performance indicators when used in media planning.

### 2. Structural Invisibility of Organic Dynamics

By construction, SKAN has no visibility into organic contribution. When multiple channels contribute to awareness or intent, SKAN ignores all but the last-clicking paid source. Organic installs are not only unmodeled—they are unobserved. This leaves practitioners blind to fluctuations in organic demand, which can be substantial during app updates, seasonal trends, or viral bursts.

### 3. Sparse and Censored Observations

SKAN reports are heavily delayed, privacy-thresholded, and aggregated. Install and revenue data are reported in coarse buckets and often only for users who exceed specific engagement thresholds. These constraints result in censored datasets that disproportionately reflect high-performing cohorts while omitting long-tail conversions. This bias is not random: it's structurally tethered to the probability of engagement, which is often correlated with campaign type, targeting, and user value. What's more, Self Reporting Networks like Meta require a limited tracking period of the first 24 hours in order to integrate with their adtech platform, which introduces noise when modeling revenue horizons at 7, 30, 90, 365 days, etc.

### 4. Opacity to Channel Interactions

SKAN treats channels independently and ignores the interactive effects of simultaneous campaigns. For example, simultaneous pushes in TikTok and ASA may influence installs synergistically or cannibalize each other. Last-touch attribution assigns revenue to the final channel without accounting for such interaction, further eroding its interpretability and planning utility.


## Model Overview

MMX aims to recover accurate estimates of paid media effectiveness while accounting for unobserved organic volume, shared latent shocks, and attribution biases such as halo and cannibalization. Unlike traditional attribution models that operate downstream of observed revenue, MMX integrates attribution, seasonality, latent demand, and media response curves into a single probabilistic system.

### 1. Unified Structure for Attribution and Revenue Estimation

The model jointly estimates:

Channel-level revenue contributions from paid media

Organic revenue trends driven by latent demand

The true aggregate revenue (used as ground truth)

SKAN-reported revenue (used as noisy attribution signal)


This joint structure allows the model to learn attribution adjustments informed by both the spend-driven media response curves and deviations between observed SKAN values and total revenue.

### 2. Latent States for Organic and Paid Demand

MMX includes a shared ARMA(3,1) latent state, capturing short-run shocks to monetization and player behavior. This shared state is partitioned multiplicatively into paid and organic components using channel-specific deviation terms (delta_paid, delta_org). These deviations account for idiosyncratic fluctuations across the revenue stack and are modeled separately to enable differential attribution and noise control.

The shared state is also modulated by a local linear trend, composed of a time-varying level and slope. This trend absorbs slow-moving changes in the game economy or user behavior that are not attributable to media or seasonal inputs.

### 3. Modeling Halo, Cannibalization, Poaching, and Censoring Effects

Unlike SKAN, MMX explicitly incorporates:

Halo effects: Paid media can increase organic installs via brand spillover or search behavior.

Cannibalization effects: Paid media may steal credit for installs that would have occurred organically.

Poaching effects: Paid media may steal credit for installs should be credited to other paid channels.

Censoring effects: Campaigns that do not meet the threshold for install volume may be censored and return no information, scales inversely with spend.

These effects are modeled directly in the SKAN likelihood, using parameterized terms informed by spend and organic volume. This structure allows MMX to disentangle true channel lift from attribution bias and helps the model reject SKAN estimates when they conflict with aggregate revenue behavior.

### 4. Media Response Functions

Paid media effects are modeled through a two-step transformation:

Adstock: Models lagged decay in user response to past spend. Usually assumed to be zero in digital marketing but can be set as a small constant like 0.1, or informed by previous experiments 

Hill function: Captures saturation and diminishing returns at higher spend levels.


Each channel has unique parameters (β, k, slope) estimated from data, allowing for customized elasticity profiles.

### 5. Events and Seasonality

MMX includes binary and continuous event covariates (e.g., holidays, playoffs, product launches) that modify the seasonality and baseline demand levels multiplicatively. These covariates are shared across both paid and organic components to reflect system-wide behavior shifts.


## Model Specification

We define a Bayesian generative model that jointly explains observed aggregate revenue ($Y_t^{\text{agg}}$) and SKAN-derived channel revenue ($Y_{t,c}^{\text{skan}}$) using latent monetization states, transformed media effects, seasonal signals, and attribution bias mechanisms. The objective is to extract accurate estimates of the underlying media response curves while reconciling observable but biased SKAN signals with aggregate outcomes.

Notation

$t = 1, \dots, T$: time index

$c = 1, \dots, C$: channel index

$S_{t,c}$: spend for channel $c$ at time $t$

$X_{t,c} \in \mathbb{R}^L$: lagged spend vector for channel $c$


---

Common Trend

The long-run monetization trend is modeled as a local linear trend:

\begin{aligned}
\mu_t &= \mu_{t-1} + \beta_{t-1} + \sigma_\mu \cdot \epsilon_t \\
\beta_t &= \beta_{t-1} + \sigma_\beta \cdot \eta_t \\
\epsilon_t, \eta_t &\sim \mathcal{N}(0, 1)
\end{aligned}

---

Shared State: Latent ARMA Process

The shared monetization signal is defined via a softplus-transformed ARMA(3,1) process:

z_t = \rho_1 z_{t-1} + \rho_2 z_{t-2} + \rho_3 z_{t-3} + \theta \cdot \epsilon_{t-1} + \epsilon_t
\quad \text{with } \epsilon_t \sim \mathcal{N}(0, \sigma_z)

\text{SharedState}_t = \text{softplus}(z_t)

---

Partitioned Monetization States

The paid and organic states are multiplicative deviations from the shared state:

\text{State}_{t}^{\text{paid}} = \text{SharedState}_t \cdot \delta_{t}^{\text{paid}}, \quad \delta_{t}^{\text{paid}} \sim \text{AR}(1)

\text{State}_{t}^{\text{org}} = \text{SharedState}_t \cdot \delta_{t}^{\text{org}}, \quad \delta_{t}^{\text{org}} \sim \mathcal{N}(\beta_{\text{trend}} \cdot \text{GoogleTrend}_t, \sigma_{\delta^{\text{org}}})

---

Media Effects: Hill-Adstock Transformation

For each channel:

f_{t,c} = \frac{1}{1 + \left( \frac{\text{Adstock}(X_{t,c})}{k_c} \right)^{-s_c}}, \quad 
\text{Adstock}(X_{t,c}) = \sum_{l=1}^L w_l \cdot X_{t,c,l}

---

Aggregate Revenue Likelihood

Y_t^{\text{agg}} \sim \mathcal{N}(\mu_t, \sigma_{\text{agg}})

\mu_t = \left( \nu \cdot \text{State}_{t}^{\text{org}} + \text{State}_{t}^{\text{paid}} \cdot \sum_{c=1}^C \beta_c f_{t,c} \right) \cdot (1 + \text{Trend}_t + \text{Season}_t)


---

SKAN Likelihood

Y_{t,c}^{\text{skan}} \sim \mathcal{N}(\mu_{t,c}^{\text{skan}}, \sigma_{\text{skan}})

\mu_{t,c}^{\text{skan}} = \text{Base}_{t,c} + \text{Cannib}_{t,c} + \text{Halo}_{t,c}

Where:

$\text{Base}{t,c} = R{t,c}^{\text{true}} \cdot \left(1 - \gamma \cdot \exp(-S_{t,c} / s)\right)$

$\text{Cannib}{t,c} = \omega_c^{\text{org}} \cdot \log(1 + \text{State}{t}^{\text{org}}) \cdot \log(1 + S_{t,c}) + \sum_{j \neq c} \omega_j^{\text{cross}} \cdot \log(1 + S_{t,j})$

$\text{Halo}{t,c} = -\phi_c \cdot \log(1 + S{t,c})$



---

Prior Specification

Latent states are regularized using tight priors on $\sigma_{\delta^{\text{paid}}}, \sigma_z$

Role-specific priors enforce structural identifiability on halo and cannibalization parameters

Shared shock terms are deactivated unless specifically enabled for posterior residual diagnostics



