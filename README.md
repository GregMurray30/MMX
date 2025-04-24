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

We define a Bayesian generative model that jointly explains observed aggregate revenue ($Y_t^{\text{agg}}$) and SKAN-derived channel revenue ($Y_{t,c}^{\text{skan}}$) using latent monetization states, transformed media effects, seasonal signals, and attribution bias mechanisms. The objective is to extract accurate estimates of the underlying media response curves while reconciling observable but biased SKAN signals with aggregate outcomes. See model_specification_math.pdf for formal model specification which is currently a work in progress.


## Simulation Strategy

To evaluate the performance of the MMX model under controlled and diagnostically informative conditions, we developed a flexible simulation framework that generates realistic, weekly-level marketing and revenue data across multiple digital channels. This framework allows us to validate whether the MMX model improves attribution over baseline methods (e.g., SKAN-derived response curves), particularly in the presence of known halo effects, cannibalization biases, and latent confounding through shared monetization states.

### Data Generation Process

We simulate media spend across four channels over a 52-week horizon, with temporal patterns informed by real-world campaign behaviors, including variable flighting, blackout periods, and bursts of isolated spend. Spend is transformed through Hill and Adstock functions to mimic diminishing returns and memory effects. These transformed signals interact multiplicatively with latent states and seasonality to produce aggregate revenue. Importantly, a local linear trend and an ARMA(3,1)-based latent monetization state drive both paid and organic components, with separate deviation terms (delta parameters) per channel type.

Seasonal events such as holidays and product launches are injected through additive effects, and Google Trends serves as a correlated covariate for the organic latent state.

### SKAN Bias Simulation

We simulate SKAN postbacks using the true channel-level revenue, modified by three sources of structured noise:

1. Censoring Bias: High spend results in more complete reporting; lower spend leads to information loss, modeled via exponential decay.


2. Cannibalization: Simulates misattribution from organic and other channels due to last-touch attribution. It is modeled as a function of the log of other channels’ spend and organic traffic volume.


3. Halo Effect: Represents under-attribution of paid media that indirectly boosts organic installs. This is simulated as a negative adjustment to SKAN postbacks, driven by log spend of the same channel.



Gaussian noise is added to SKAN observations, completing the distortion relative to true channel revenue.

### Evaluation Design

For each simulation:

We train MMX and a benchmark SKAN-derived response curve model on the same synthetic data.

The benchmark model uses a Bayesian framework with fixed event controls and a linear trend index, but no latent states or SKAN de-biasing.

For each channel, we compute Mean Absolute Percentage Error (MAPE) against the known true response curve.

Performance is compared across MMX and SKAN using directional accuracy, percentage of simulations where MMX outperforms, and diagnostic correlation analyses.


### Scope of Simulations

We conduct 100 simulations across a spectrum of latent state correlations (15%–40%), halo and cannibalization intensities, and SKAN noise levels. Simulations are stratified by conditions such as channel isolation, collinearity of spend, and total scale, enabling generalization of findings.


## Results

Across 100 simulation runs, the MMX model consistently demonstrated an improvement over SKAN-derived response curves in the presence of structured attribution bias. The results are evaluated on three channels with varying characteristics, anonymized as Channel A, Channel B, and Channel C.

### Overall Performance

Channel A (High spend, low correlation with other channels):
MMX outperformed SKAN in 93% of simulations. Performance was robust across both high and low halo/cannibalization regimes. This channel’s low collinearity and high scale facilitated identifiability of its causal impact.

Channel B (Low total spend, high co-spend with other channels):
MMX outperformed SKAN in 70% of simulations. Performance improved when cannibalization priors were loosened and the channel had short periods of isolated spend, suggesting identifiability remains contingent on data structure.

Channel C (Moderate spend, moderate co-spend overlap):
MMX outperformed SKAN in 54% of simulations. Results were sensitive to latent state correlation and prior specification. In simulations where SKAN biases were large (e.g., halo magnitude > 1.0), MMX provided significant directional improvement. In low-bias or tightly coupled simulations, MMX occasionally diverged incorrectly.


### Attribution Error Analysis

MMX reduced attribution error by an average of 22% on Channel A and 11% on Channel B, while performance on Channel C was net neutral.

When SKAN attribution was highly distorted (e.g., combined halo + cannibalization bias exceeding 30% of channel revenue), MMX reduced net attribution error in >80% of such cases.

In cases where MMX underperformed, most errors stemmed from organic-predictive entanglement (e.g., misalignment between organic latent states and revenue), or from insufficient differentiation in spend patterns between channels.


### Simulation Diagnostics

We found that MMX's ability to outperform SKAN was strongly associated with:

Lower inter-channel spend correlation (ASA-like patterns)

Greater scale of spend

Wider priors on halo and cannibalization weights

Latent state correlation between paid and organic in the 20–30% range


These findings validate the model’s robustness in realistic settings, while also highlighting limitations in regimes of low identifiability.





