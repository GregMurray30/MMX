

functions {

  real Hill(real t, real k, real slope) {

    return 1 / (1 + pow(t / k, -slope));

  }

 

  real Adstock(row_vector t, row_vector weights) {

    return dot_product(t, weights);

  }

 

  real softplus(real x) {

    return log1p_exp(x);

  }

}

 

data {

  int<lower=1> N_train;

  int<lower=1> num_media;

  int<lower=1> max_lag;

 

  vector[N_train] Y_true_train;

  matrix[N_train, num_media] Y_skan_train;

  int<lower=0, upper=1> skan_mask[N_train, num_media];

  matrix[N_train, num_media] spend_train;

  row_vector[max_lag] X_media_train[N_train, num_media];

 

  row_vector[N_train] week_train;

  real prd;

 

  row_vector[N_train] google_trends_train;

 

  row_vector[N_train] is_holiday_train;

  row_vector[N_train] playoffs_1st_train;

  row_vector[N_train] playoffs_2nd_train;

  row_vector[N_train] playoffs_3rd_train;

  row_vector[N_train] playoffs_finals_train;

  row_vector[N_train] allstar_train;

  row_vector[N_train] ssn_launch_train;

  row_vector[N_train] olympics_train;

  row_vector[N_train] curry_pckg_train;

 

  real<lower=0> spend_scale;

  int<lower=-1, upper=1> cannibalization_role[num_media];

  int<lower=0, upper=1> halo_role[num_media];

 

    //Channel priors

  vector[num_media] beta_prior_mns;

  vector[num_media] beta_prior_sds;

  vector[num_media] k_prior_mns;

  vector[num_media] k_prior_sds;

  vector[num_media] slope_prior_mns;

  vector[num_media] slope_prior_sds;

 

  real beta_holiday_mn;

  real beta_holiday_sd;

  real beta_playoffs_1st_mn;

  real beta_playoffs_1st_sd;

  real beta_playoffs_2nd_mn;

  real beta_playoffs_2nd_sd;

  real beta_playoffs_3rd_mn;

  real beta_playoffs_3rd_sd;

  real beta_playoffs_finals_mn;

  real beta_playoffs_finals_sd;

  real beta_allstar_mn;

  real beta_allstar_sd;

  real beta_ssn_launch_mn;

  real beta_ssn_launch_sd;

  real beta_olympics_mn;

  real beta_olympics_sd;

  real beta_curry_pckg_mn;

  real beta_curry_pckg_sd;




  // Prior means and standard deviations for scalars

  // Vectors for priors on channel-level noise parameters

  //vector[num_media] channel_noise_prior_mn;

// vector[num_media] channel_noise_prior_sd;

 

  real sigma_skan_mn;

  real sigma_skan_sd;

  vector[num_media] cannibalization_effect_mn;

  vector[num_media] cannibalization_effect_sd;

  real gamma_censor_alpha;

  real gamma_censor_beta;

  vector[num_media] organic_cannib_wt_mn;

  vector[num_media] organic_cannib_wt_sd;

  vector[num_media] organic_halo_wt_mn;

  vector[num_media] organic_halo_wt_sd;




  real sigma_mult_prior_mn;

  real sigma_mult_prior_sd;

  real sigma_add_prior_mn;

  real sigma_add_prior_sd;

 

 

  // ARMA state priors

  real sigma_state_shared_mn;

  real sigma_state_shared_sd;

  real state_shared_raw_mn;

  real state_shared_raw_sd;

 

  real sigma_state_dev_org_sd;

  real sigma_state_dev_paid_sd;

 

  // ARMA coefficient priors

  real rho1_mn;

  real rho1_sd;

  real rho2_mn;

  real rho2_sd;

  real rho3_mn;

  real rho3_sd;

  real theta_mn;

  real theta_sd;

 

  real rho_paid_mn;

  real rho_paid_sd;





  // Stage 2 parameters become data

 




// Local trend priors

  real mu0_trend_mn;

  real mu0_trend_sd;

  real beta0_trend_mn;

  real beta0_trend_sd;

  real epsilon_trend_mn;

  real epsilon_trend_sd;

  real sigma_level_trend_mn;

  real sigma_level_trend_sd;

  real sigma_slope_trend_mn;

  real sigma_slope_trend_sd;

  real eta_trend_mn;

  real eta_trend_sd;






}

 

transformed data{

  int T= N_train+0;

 

}

 

parameters {

 

    real<lower=0> nu;

 

   // real shared_shock_scale; // scalar multiplier between 0 and 1

 

   real<lower=0> beta_holiday;

  real<lower=0> beta_playoffs_1st;

  real<lower=0> beta_playoffs_2nd;

  real<lower=0> beta_playoffs_3rd;

  real<lower=0> beta_playoffs_finals;

  real<lower=0> beta_allstar;

  real<lower=0> beta_curry_pckg;

  real<lower=0> beta_ssn_launch;

  real<lower=0> beta_olympics;

 

  vector<lower=0>[num_media] beta_medias;

  vector<lower=1>[num_media] k;

  vector<lower=0.95, upper=4>[num_media] slope;




  real<lower=-.25, upper=.6> rho1;

  real<lower=-.25, upper=.3> rho2;

  real<lower=-.25, upper=.2> rho3;

  real<lower=-.25, upper=.25> theta;

 

  real<lower=-.25, upper=.8> rho_paid;

 

  real<lower=0> sigma_add;

// real<lower=0> sigma_mult;

 

  real mu0_trend;

  real beta0_trend;

  real<lower=0> sigma_level_trend;

  real<lower=0> sigma_slope_trend;

  vector[T] epsilon_trend;

  vector[T] eta_trend;




  real<lower=0> sigma_skan;

  real<lower=0, upper=1> gamma_censor;

  vector<lower=0, upper=1>[num_media] cannibalization_effect;

  vector<lower=0>[num_media] organic_cannib_wt;

  vector<lower=0>[num_media] organic_halo_wt;

 

  //real<lower=0, upper=1> shared_shock_scale; // scalar multiplier between 0 and 1

  vector[T] shared_shock_raw; // raw shared shock (unit normal)

 

  real<lower=0> beta_google_trend;

 

  real<lower=0> sigma_state_shared;

  real<lower=0> sigma_state_dev_org;

  real<lower=0> sigma_state_dev_paid;

  vector[N_train] state_shared_raw;

  vector<lower=0>[N_train] delta_paid;

  vector<lower=0>[N_train] delta_org;






}

 

transformed parameters {

 

 

  // 1. Compute lag weights (as in V2)

  row_vector[max_lag] lag_weights;

  for (lag in 1:max_lag)

    lag_weights[lag] = 1;

   

  // 2. Compute additional diagnostic matrices (as in V2)

  matrix[N_train, num_media] cum_effects_hill_train;

  matrix[N_train, num_media] censor_bias_train;

  vector[N_train] mu_train;

  matrix[N_train, num_media] R_true_channel_train;

 

  vector[N_train] state_paid;

  vector[N_train] state_org;




 

  // For each training observation, compute the Hill-transformed media effect and censor bias:

  for (n in 1:N_train) {

    for (c in 1:num_media) {

      real cum_effect = Adstock(X_media_train[n, c], lag_weights);

      cum_effects_hill_train[n, c] = Hill(cum_effect, k[c], slope[c]);

      censor_bias_train[n, c] = gamma_censor * exp(-spend_train[n, c] / spend_scale);

    }

  }

 

 

    // 3. Compute a Common Trend Component (shared by ORG and PAID)

  // Here we implement a simple local linear trend as a cumulative sum of a constant (no additional noise)

  // since you mentioned no sigma_trend parameter in your model.

  vector[T] level_trend; // The evolving level of the common trend

  vector[T] slope_trend; // The evolving slope (trend) of the common trend

  vector[T] common_trend; // The final common trend signal

 

  level_trend[1] = mu0_trend; // mu0_trend is defined in the parameters

  slope_trend[1] = beta0_trend; // beta0_trend is defined in the parameters

  // For t>=2, update the trend components. (No additional noise is added here.)

  for (t in 2:T) {

    slope_trend[t] = slope_trend[t-1] + sigma_slope_trend * eta_trend[t]; // Constant slope (can change if you add noise)

    level_trend[t] = level_trend[t-1] + slope_trend[t-1] + sigma_level_trend * epsilon_trend[t];

  }

  // The common trend is simply the level:

  common_trend = level_trend;

 

  vector[T] common_trend_nrm = common_trend - mean(common_trend) + 1;

 

// 4. Compute the ARMA (short-run) components for shared

 

  vector[T] state_shared_combined;

 

  // Initialize the first 3 time points centered at 1

  state_shared_combined[1] = 1 + sigma_state_shared * state_shared_raw[1];

  state_shared_combined[2] = 1 + rho1 * (state_shared_combined[1] - 1)

                                + sigma_state_shared * state_shared_raw[2];

  state_shared_combined[3] = 1 + rho1 * (state_shared_combined[2] - 1)

                                + rho2 * (state_shared_combined[1] - 1)

                                + sigma_state_shared * state_shared_raw[3];

 

  // Recursion for t = 4 to T

  for (t in 4:T) {

    state_shared_combined[t] = 1

      + rho1 * (state_shared_combined[t-1] - 1)

      + rho2 * (state_shared_combined[t-2] - 1)

      + rho3 * (state_shared_combined[t-3] - 1)

      + theta * (state_shared_combined[t-1] - 1)

      + sigma_state_shared * state_shared_raw[t];

  }

 

  // 5. Partition the latent states into training and holdout sets

  vector[N_train] state_train_shared = state_shared_combined[1:N_train];

 

  // 6. Partition into paid and organic states

  for (n in 1:N_train) {

    state_paid[n] = state_train_shared[n] * delta_paid[n];

    state_org[n] = state_train_shared[n] * delta_org[n];

  }

 

  // 8. Now compute mu_train and R_true_channel_train using the new latent states.

  // For each training point:

  for (n in 1:N_train) {

    real season_trend =

      common_trend[n] +

      beta_holiday * is_holiday_train[n] +

      beta_playoffs_1st * playoffs_1st_train[n] +

      beta_playoffs_2nd * playoffs_2nd_train[n] +

      beta_playoffs_3rd * playoffs_3rd_train[n] +

      beta_playoffs_finals * playoffs_finals_train[n] +

      beta_allstar * allstar_train[n] +

      beta_ssn_launch * ssn_launch_train[n] +

      beta_olympics * olympics_train[n] +

      beta_curry_pckg * curry_pckg_train[n];

    real channel_sum = dot_product(beta_medias, cum_effects_hill_train[n]);

    mu_train[n] =  ( state_org[n] * nu + state_paid[n] * channel_sum ) * season_trend;

 

    for (c in 1:num_media) {

      R_true_channel_train[n, c] = state_paid[n] *

                                   beta_medias[c] * cum_effects_hill_train[n, c] * season_trend;

    }

 

  }

 

  real<lower=0> sigma_shared_max = sd(.2*(mu_train-Y_true_train));

 

  vector[T] shared_shock = shared_shock_raw * 0 * sigma_shared_max;

 

}




model {

  nu ~ normal(20, 5);

 

  sigma_add ~ normal(sigma_mult_prior_mn, sigma_mult_prior_sd);

// sigma_mult ~ normal(sigma_add_prior_mn, sigma_add_prior_sd);

 

  beta_holiday ~ normal(beta_holiday_mn, beta_holiday_sd);

  beta_playoffs_1st ~ normal(beta_playoffs_1st_mn, beta_playoffs_1st_sd);

  beta_playoffs_2nd ~ normal(beta_playoffs_2nd_mn, beta_playoffs_2nd_sd);

  beta_playoffs_3rd ~ normal(beta_playoffs_3rd_mn, beta_playoffs_3rd_sd);

  beta_playoffs_finals ~ normal(beta_playoffs_finals_mn, beta_playoffs_finals_sd);

  beta_allstar ~ normal(beta_allstar_mn, beta_allstar_sd);

  beta_ssn_launch ~ normal(beta_ssn_launch_mn, beta_ssn_launch_sd);

  beta_olympics ~ normal(beta_olympics_mn, beta_olympics_sd);

  beta_curry_pckg ~ normal(beta_curry_pckg_mn, beta_curry_pckg_sd);

 

  beta_medias ~ normal(beta_prior_mns, beta_prior_sds);

   k ~ normal(k_prior_mns, k_prior_sds);

   slope ~ normal(slope_prior_mns, slope_prior_sds);

 

  organic_cannib_wt[1] ~ normal(organic_cannib_wt_mn[1], organic_cannib_wt_sd[1]);

  organic_cannib_wt[2] ~ normal(organic_cannib_wt_mn[2], organic_cannib_wt_sd[2]);

  organic_cannib_wt[3] ~ normal(organic_cannib_wt_mn[3], organic_cannib_wt_sd[3]);

  organic_cannib_wt[4] ~ normal(organic_cannib_wt_mn[4], organic_cannib_wt_sd[4]);

 

  organic_halo_wt[1] ~ normal(organic_halo_wt_mn[1], organic_halo_wt_sd[1]);

  organic_halo_wt[2] ~ normal(organic_halo_wt_mn[2], organic_halo_wt_sd[2]);

  organic_halo_wt[3] ~ normal(organic_halo_wt_mn[3], organic_halo_wt_sd[3]);

  organic_halo_wt[4] ~ normal(organic_halo_wt_mn[4], organic_halo_wt_sd[4]);

 

  mu0_trend ~ normal(mu0_trend_mn, mu0_trend_sd);

  beta0_trend ~ normal(beta0_trend_mn, beta0_trend_sd);

  epsilon_trend ~ normal(epsilon_trend_mn, epsilon_trend_sd);

  sigma_level_trend ~ normal(sigma_level_trend_mn, sigma_level_trend_sd);

  sigma_slope_trend ~ normal(sigma_slope_trend_mn, sigma_slope_trend_sd);

  eta_trend ~ normal(eta_trend_mn, eta_trend_sd);

 

  beta_google_trend ~ normal(0, .5);

  sigma_state_shared ~ normal(sigma_state_shared_mn, sigma_state_shared_sd);

  state_shared_raw ~ normal(state_shared_raw_mn, state_shared_raw_sd);

 

  sigma_state_dev_org ~ normal(0, sigma_state_dev_org_sd);

  sigma_state_dev_paid ~ normal(0, sigma_state_dev_paid_sd);

 

  for (n in 1:N_train) {

    delta_org[n] ~ normal(1 + beta_google_trend * google_trends_train[n], sigma_state_dev_org);

  }

 

  delta_paid[1] ~ normal(1, sigma_state_dev_paid);

  for (n in 2:N_train){

    delta_paid[n] ~ normal(1 + rho_paid * (delta_paid[n-1]-1), sigma_state_dev_paid);

  }




  rho1 ~ normal(rho1_mn, rho1_sd);

  rho2 ~ normal(rho2_mn, rho2_sd);

  rho3 ~ normal(rho3_mn, rho3_sd);

  theta ~ normal(theta_mn, theta_sd);

  rho_paid ~ normal(rho_paid_mn, rho_paid_sd);

 

  sigma_skan ~ normal(sigma_skan_mn, sigma_skan_sd);

  gamma_censor ~ beta(gamma_censor_alpha, gamma_censor_beta);

 

  for (c in 1:num_media)

    cannibalization_effect[c] ~ normal(cannibalization_effect_mn[c] * cannibalization_role[c], cannibalization_effect_sd[c]);

 

 

  for (n in 1:N_train) {

    Y_true_train[n] ~ normal(mu_train[n] + shared_shock[n], fmax(1e-5, sigma_add  )); #additive noise since we have latent states for heteroscedasticity

   // Y_true_train[n] ~ normal(mu_train[n] + shared_shock[n], sigma_add + softplus((mu_train[n]+.01)*sigma_mult));

 

    for (c in 1:num_media) {

      real base_skan = R_true_channel_train[n, c] * (1 - censor_bias_train[n, c]);

      real cannib_term = 0;

      real halo_term = 0;

      for (j in 1:num_media) {

        if (j != c) {

          cannib_term += cannibalization_effect[c] * log1p(fmax(spend_train[n, j], 1e-2)) * log1p(fmax(spend_train[n, c], 1e-2));

        }

      }

 

      cannib_term += organic_cannib_wt[c] * cannibalization_role[c] * log1p(fmax(spend_train[n, c], 1e-2)) * log1p(fmax(state_org[n], 1e-2));

      halo_term += -1 * organic_halo_wt[c] * halo_role[c] * log1p(fmax(spend_train[n, c], 1e-2));

      Y_skan_train[n, c] ~ normal(base_skan + cannib_term + halo_term + shared_shock[n], sigma_skan);

    }

  }

}

 

generated quantities {

  vector[N_train] Y_true_pred;

  matrix[N_train, num_media] Y_skan_pred;

  vector[N_train] season_trend_train;

  vector[N_train] channel_sum_train;

  vector[N_train] organics_train;

  // Optionally, include the common trend as diagnostics:

  vector[N_train] common_trend_train;

 

  matrix[N_train, num_media] cannib_term_train;

  matrix[N_train, num_media] halo_term_train;

 

  // Process training set predictions:

  for (n in 1:N_train) {

    common_trend_train[n] = common_trend[n];

 

    // Compute the seasonal/event multiplier using the seasonal coefficients:

    real season_trend = common_trend_train[n] +

      beta_holiday * is_holiday_train[n] +

      beta_playoffs_1st * playoffs_1st_train[n] +

      beta_playoffs_2nd * playoffs_2nd_train[n] +

      beta_playoffs_3rd * playoffs_3rd_train[n] +

      beta_playoffs_finals * playoffs_finals_train[n] +

      beta_allstar * allstar_train[n] +

      beta_ssn_launch * ssn_launch_train[n] +

      beta_olympics * olympics_train[n] +

      beta_curry_pckg * curry_pckg_train[n];

    season_trend_train[n] = season_trend;

   

    // Compute the channel sum for media inputs (using the Hill-adstock transformation):

    channel_sum_train[n] = dot_product(beta_medias, cum_effects_hill_train[n]);

   

    // Compute the organic component:

    organics_train[n] = state_org[n] * nu ;

   

    // Compute overall revenue prediction using the latent state:

    Y_true_pred[n] = ( organics_train[n] + state_paid[n] * channel_sum_train[n] ) * season_trend;

   

    

    // For attribution: SKAN predictions (using R_true_channel_train computed in the transformed block)

    for (c in 1:num_media) {

      real base_skan = R_true_channel_train[n, c] * (1 - censor_bias_train[n, c]);

      real cannib_term = 0;

      real halo_term = 0;

      for (j in 1:num_media) {

        if (j != c) {

          cannib_term += cannibalization_effect[c] * log1p(fmax(spend_train[n, j], 1e-2)) * log1p(fmax(spend_train[n, c], 1e-2));

        }

      }

     

      cannib_term += organic_cannib_wt[c] * cannibalization_role[c] * log1p(fmax(spend_train[n, c], 1e-2))  * log1p(fmax(state_org[n], 1e-2));

      halo_term += -1 * organic_halo_wt[c] * halo_role[c] * log1p(fmax(spend_train[n, c], 1e-2));

      Y_skan_pred[n, c] = fmax(0, base_skan + cannib_term + halo_term + shared_shock[n] );

 

      cannib_term_train[n, c] = cannib_term;

      halo_term_train[n, c] = halo_term;  

    }

   

    // Output the common trend component (for diagnostic purposes)

   

    // Optionally, compute Fourier seasonality if your model uses it:

    // fourier_ssn_train[n] = beta_day_x * cos(2 * pi() * week_train[n] / prd)

    // + beta_day_y * sin(2 * pi() * week_train[n] / prd);

  }

    
}
