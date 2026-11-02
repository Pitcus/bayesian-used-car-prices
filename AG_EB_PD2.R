# ============================================================
# Projekt: Bayesowska analiza cen używanych samochodów
# Praca domowa 2 - Model z rozkładem t składnika losowego
# ============================================================

library(rstan)
install.packages("coda")
library(coda)
install.packages("bridgesampling")
library(bridgesampling)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

setwd("/Users/antonigolebiewski/Desktop/Ekonometria Bayesowska/PD2")

# Wczytanie próby z PD1
data_model <- read.csv("used_cars_sample_300.csv")

# ============================================================
# 2. Przygotowanie macierzy X i wektora y
# ============================================================

model_formula <- log_price ~ Age + Milage_1000 + Engine_L +
  Accident + Automatic + Hybrid + Diesel

y <- as.numeric(data_model$log_price)
X <- model.matrix(model_formula, data = data_model)

N <- nrow(X)
k <- ncol(X)

N
k
colnames(X)

# ============================================================
# 3. Parametry priorów
# ============================================================

beta_prior <- c(
  10.2941,   # Intercept
  -0.05,     # Age
  -0.005,    # Milage_1000
  0.10,      # Engine_L
  -0.15,     # Accident
  0.00,      # Automatic
  0.10,      # Hybrid
  0.05       # Diesel
)

beta_sd_prior <- c(
  2.0,    # Intercept
  0.03,   # Age
  0.003,  # Milage_1000
  0.08,   # Engine_L
  0.15,   # Accident
  0.15,   # Automatic
  0.20,   # Hybrid
  0.20    # Diesel
)

# ============================================================
# 4. Estymacja modelu w Stanie
# ============================================================

stan_data <- list(
  N = N,
  k = k,
  y = y,
  X = X,
  beta_prior = beta_prior,
  beta_sd_prior = beta_sd_prior
)

fit <- stan(
  file = "AG_EB_PD2_model.stan",
  data = stan_data,
  iter = 4000,
  warmup = 1000,
  chains = 4,
  seed = 123
)

print(fit, pars = c("beta", "sigma", "nu"))

# ============================================================
# 5. Wyciągnięcie próbek a posteriori
# ============================================================

posterior_samples <- extract(fit)

# Średnie a posteriori
beta_post_mean <- apply(posterior_samples$beta, 2, mean)
sigma_post_mean <- mean(posterior_samples$sigma)
nu_post_mean <- mean(posterior_samples$nu)

names(beta_post_mean) <- colnames(X)

beta_post_mean
sigma_post_mean
nu_post_mean

# ============================================================
# 6. Tabela porównawcza PD1 vs PD2
# ============================================================

beta_post_sd <- apply(posterior_samples$beta, 2, sd)
names(beta_post_sd) <- colnames(X)

# Wartości z PD1 (z raportu)
beta_prior_vec <- c(10.2941, -0.05, -0.005, 0.10, -0.15, 0.00, 0.10, 0.05)
beta_PD1 <- c(10.6512, -0.0541, -0.0078, 0.1853, -0.0783, -0.0362, 0.0671, 0.1830)

posterior_table <- data.frame(
  Parameter     = colnames(X),
  Prior_Mean    = beta_prior_vec,
  PD1_Post_Mean = beta_PD1,
  PD2_Post_Mean = round(beta_post_mean, 4),
  PD2_Post_SD   = round(beta_post_sd, 4)
)

print(posterior_table)

# ============================================================
# 7. Przedziały HPDI 95%
# ============================================================

beta_hpdi <- HPDinterval(as.mcmc(posterior_samples$beta), prob = 0.95)
rownames(beta_hpdi) <- colnames(X)

hpdi_table <- data.frame(
  Parameter  = colnames(X),
  Post_Mean  = round(beta_post_mean, 4),
  HPDI_Lower = round(beta_hpdi[, 1], 4),
  HPDI_Upper = round(beta_hpdi[, 2], 4)
)

hpdi_table$Contains_Zero <- 
  hpdi_table$HPDI_Lower <= 0 & hpdi_table$HPDI_Upper >= 0

print(hpdi_table)

# ============================================================
# 8. Diagnostyka - trace plots
# ============================================================

rstan::traceplot(fit, pars = c("beta", "sigma", "nu"), 
                 inc_warmup = FALSE,
                 nrow = 5)

# ============================================================
# 9. Rozkłady brzegowe a posteriori
# ============================================================

if (!dir.exists("plots")) dir.create("plots")

param_names <- colnames(X)

for (j in 1:k) {
  png(
    filename = paste0("plots/posterior_", j, "_", 
                      gsub("[^A-Za-z0-9]", "", param_names[j]), ".png"),
    width = 900, height = 600
  )
  
  samples_j <- posterior_samples$beta[, j]
  
  plot(density(samples_j),
       main = paste("Rozkład brzegowy a posteriori:", param_names[j]),
       xlab = param_names[j],
       ylab = "Gęstość",
       lwd = 2)
  
  abline(v = mean(samples_j), lty = 2, lwd = 2)
  
  dev.off()
}

# Dodatkowo sigma i nu
png("plots/posterior_sigma.png", width = 900, height = 600)
plot(density(posterior_samples$sigma),
     main = "Rozkład brzegowy a posteriori: sigma",
     xlab = "sigma", ylab = "Gęstość", lwd = 2)
abline(v = sigma_post_mean, lty = 2, lwd = 2)
dev.off()

png("plots/posterior_nu.png", width = 900, height = 600)
plot(density(posterior_samples$nu),
     main = "Rozkład brzegowy a posteriori: nu",
     xlab = "nu", ylab = "Gęstość", lwd = 2)
abline(v = nu_post_mean, lty = 2, lwd = 2)
dev.off()

# ============================================================
# 10. Prognoza punktowa i przedziałowa
# ============================================================

# Ta sama obserwacja co w PD1
obs_id <- 1
x_tau <- matrix(X[obs_id, ], nrow = 1)

# Próbki y_pred dla tej obserwacji z bloku generated quantities
y_pred_samples <- posterior_samples$y_pred[, obs_id]

# Prognoza punktowa i przedział predykcyjny
forecast_log  <- mean(y_pred_samples)
forecast_lower <- quantile(y_pred_samples, 0.025)
forecast_upper <- quantile(y_pred_samples, 0.975)

# Przekształcenie do skali ceny
forecast_price       <- exp(forecast_log)
forecast_price_lower <- exp(forecast_lower)
forecast_price_upper <- exp(forecast_upper)

actual_price <- data_model$price_num[obs_id]

prediction_result <- data.frame(
  actual_price         = actual_price,
  forecast_price       = round(forecast_price, 0),
  interval_lower_95    = round(forecast_price_lower, 0),
  interval_upper_95    = round(forecast_price_upper, 0)
)

print(prediction_result)

# ============================================================
# 11. Czynniki Bayesa przez bridge sampling
# ============================================================

# Model pełny M1
bridge_M1 <- bridge_sampler(fit, silent = TRUE)
print(bridge_M1)

# Modele zredukowane M2 - bez jednej zmiennej
zmienne <- c("Age", "Milage_1000", "Engine_L", 
             "Accident", "Automatic", "Hybrid", "Diesel")

log_BF_results <- data.frame(
  zmienna = zmienne,
  log_BF  = NA_real_
)

for (i in 1:length(zmienne)) {
  
  usun <- zmienne[i]
  cols <- which(colnames(X) != usun)
  
  X_reduced    <- X[, cols]
  k_reduced    <- ncol(X_reduced)
  beta_prior_r <- beta_prior[cols]
  beta_sd_r    <- beta_sd_prior[cols]
  
  stan_data_r <- list(
    N              = N,
    k              = k_reduced,
    y              = y,
    X              = X_reduced,
    beta_prior     = beta_prior_r,
    beta_sd_prior  = beta_sd_r
  )
  
  fit_r <- stan(
    file   = "AG_EB_PD2_model.stan",
    data   = stan_data_r,
    iter   = 4000,
    warmup = 1000,
    chains = 4,
    seed   = 123,
    refresh = 0  # wycisza output
  )
  
  bridge_r <- bridge_sampler(fit_r, silent = TRUE)
  
  log_BF_results$log_BF[i] <- bridge_M1$logml - bridge_r$logml
  
  cat("Gotowe:", usun, "| log BF =", log_BF_results$log_BF[i], "\n")
}

# Tabela czynników Bayesa
log_BF_results$BF <- exp(log_BF_results$log_BF)

log_BF_results$interpretacja <- ifelse(
  log_BF_results$BF < 1,
  "dane wspierają model bez tej zmiennej",
  ifelse(
    log_BF_results$BF < 3,
    "słabe wsparcie dla zmiennej",
    ifelse(
      log_BF_results$BF < 20,
      "umiarkowane wsparcie dla zmiennej",
      ifelse(
        log_BF_results$BF < 150,
        "silne wsparcie dla zmiennej",
        "bardzo silne wsparcie dla zmiennej"
      )
    )
  )
)

print(log_BF_results)

list.files("plots")
