data {
  int<lower=0> N;          // liczba obserwacji
  int<lower=0> k;          // liczba kolumn macierzy X (łącznie ze stałą)
  vector[N] y;             // log(cena) - zmienna objaśniana
  matrix[N, k] X;          // macierz regresorów (z kolumną jedynek na stałą)
  vector[k] beta_prior;    // wartości oczekiwane priorów dla beta
  vector[k] beta_sd_prior; // odchylenia standardowe priorów dla beta
}

parameters {
  vector[k] beta;        // współczynniki regresji
  real<lower=0> sigma;   // odchylenie standardowe błędu
  real<lower=1> nu;      // stopnie swobody rozkładu t
}

model {
  beta ~ normal(beta_prior, beta_sd_prior);    // Priory dla beta
  sigma ~ normal(0, 1);   // Prior dla sigma
  nu ~ gamma(2, 0.1);   // Prior dla nu
  y ~ student_t(nu, X * beta, sigma);   // Likelihood
}

generated quantities {
  vector[N] y_pred;
  vector[N] log_lik;
  
  for (n in 1:N) {
    y_pred[n] = student_t_rng(nu, X[n] * beta, sigma);
    log_lik[n] = student_t_lpdf(y[n] | nu, X[n] * beta, sigma);
  }
}
