# ============================================================
# Projekt: Bayesowska analiza cen używanych samochodów
# 1. Wczytanie danych i przygotowanie modelu
# ============================================================

setwd('/Users/antonigolebiewski/Desktop/Ekonometria Bayesowska')
data_raw <- read.csv("used_cars.csv")

str(data_raw)
head(data_raw)
summary(data_raw)
names(data_raw)

# ============================================================
# 2. Przygotowanie zmiennych
# ============================================================

data <- data_raw

# Cena: usuwamy znaki $, przecinki i inne znaki nieliczbowe
data$price_num <- as.numeric(gsub("[^0-9.]", "", data$price))

# Przebieg: usuwamy "mi.", przecinki i inne znaki nieliczbowe
data$milage_num <- as.numeric(gsub("[^0-9.]", "", data$milage))

# Logarytm ceny - zmienna objaśniana
data$log_price <- log(data$price_num)

# Wiek auta: względem najnowszego rocznika w danych
current_year <- max(data$model_year, na.rm = TRUE)
data$Age <- current_year - data$model_year

# Przebieg w tysiącach mil
data$Milage_1000 <- data$milage_num / 1000

# Wyciągnięcie pojemności silnika z kolumny engine
# Szukamy liczby stojącej przed literą L, np. 2.0L, 3.5L
extract_engine_liters <- function(x) {
  out <- rep(NA_real_, length(x))
  matches <- regexpr("[0-9]+\\.?[0-9]*L", x)
  found <- matches != -1
  extracted <- regmatches(x, matches)
  out[found] <- as.numeric(gsub("L", "", extracted))
  return(out)
}

data$Engine_L <- extract_engine_liters(data$engine)
summary(data$Engine_L)

# ============================================================
# 3. Zmienne zero-jedynkowe
# ============================================================

# Wypadek / uszkodzenie
data$Accident <- ifelse(
  data$accident == "At least 1 accident or damage reported",
  1, 0
)

# Automatyczna skrzynia biegów.
# Uznajemy jako automatic wszystkie wpisy zawierające A/T, Automatic, CVT lub Dual Shift
data$Automatic <- ifelse(
  grepl("A/T|Automatic|CVT|Dual Shift", data$transmission, ignore.case = TRUE),
  1, 0
)

# Typ paliwa
data$Hybrid <- ifelse(data$fuel_type %in% c("Hybrid", "Plug-In Hybrid"), 1, 0)
data$Diesel <- ifelse(data$fuel_type == "Diesel", 1, 0)

# ============================================================
# 4. Wybór zmiennych do modelu i usunięcie braków
# ============================================================

data_clean <- data[, c(
  "log_price",
  "price_num",
  "Age",
  "Milage_1000",
  "Engine_L",
  "Accident",
  "Automatic",
  "Hybrid",
  "Diesel",
  "brand",
  "model",
  "model_year",
  "milage",
  "fuel_type",
  "engine",
  "transmission",
  "accident",
  "clean_title",
  "price"
)]

data_clean <- na.omit(data_clean)

nrow(data_clean)

summary(data_clean[, c(
  "price_num",
  "log_price",
  "Age",
  "Milage_1000",
  "Engine_L",
  "Accident",
  "Automatic",
  "Hybrid",
  "Diesel"
)])

table(data_clean$Accident)
table(data_clean$Automatic)
table(data_clean$Hybrid)
table(data_clean$Diesel)

# ============================================================
# 5. Ograniczenie próby do N = 300
# ============================================================

set.seed(123)

data_model <- data_clean[sample(1:nrow(data_clean), 300), ]

N <- nrow(data_model)
N

# ============================================================
# 6. Model regresji liniowej
# ============================================================

model_formula <- log_price ~ Age + Milage_1000 + Engine_L +
  Accident + Automatic + Hybrid + Diesel

# Klasyczny model pomocniczy tylko do kontroli danych,
# nie jest podstawą elicytacji a priori w poprawionej wersji.
ols_model <- lm(model_formula, data = data_model)
summary(ols_model)

# ============================================================
# 7. Macierz X i wektor y do analizy bayesowskiej
# ============================================================

y <- as.matrix(data_model$log_price)
X <- model.matrix(model_formula, data = data_model)

N <- nrow(X)
k <- ncol(X)

N
k
colnames(X)

# Liczba zmiennych objaśniających bez stałej
number_of_explanatory_variables <- k - 1
number_of_explanatory_variables
0.1 * N

if (N >= 30 & N <= 500 &
    number_of_explanatory_variables >= 2 &
    number_of_explanatory_variables <= 0.1 * N) {
  print("Warunki projektu są spełnione.")
} else {
  print("Warunki projektu NIE są spełnione.")
}

# ============================================================
# 8. Zapis próby głównej
# ============================================================

write.csv(data_model, "used_cars_sample_300.csv", row.names = FALSE)

# ============================================================
# 9. Elicytacja parametrów a priori
# ============================================================

# W poprawionej wersji prior nie jest oparty na podzbiorze danych

# ------------------------------------------------------------
# 9.1. Wartości oczekiwane a priori parametrów beta
# ------------------------------------------------------------

# Interpretacja:
# Intercept      - bazowy poziom logarytmu ceny.
# Age            - oczekiwany spadek ceny o ok. 5% za dodatkowy rok.
# Milage_1000    - oczekiwany spadek ceny o ok. 0.5% za dodatkowe 1000 mil.
# Engine_L       - dodatni wpływ większej pojemności / wyższego segmentu auta.
# Accident       - oczekiwany spadek ceny dla aut z historią wypadku.
# Automatic      - brak silnego założenia co do kierunku wpływu.
# Hybrid         - niewielka dodatnia premia za napęd hybrydowy.
# Diesel         - niewielka dodatnia różnica dla diesla.

b_prior <- matrix(c(
  mean(data_model$log_price), # Intercept
  -0.05,                      # Age
  -0.005,                     # Milage_1000
  0.10,                      # Engine_L
  -0.15,                      # Accident
  0.00,                      # Automatic
  0.10,                      # Hybrid
  0.05                       # Diesel
), ncol = 1)

rownames(b_prior) <- colnames(X)

# ------------------------------------------------------------
# 9.2. Niepewność a priori parametrów beta
# ------------------------------------------------------------

prior_sd <- c(
  2.0,    # Intercept
  0.03,   # Age
  0.003,  # Milage_1000
  0.08,   # Engine_L
  0.15,   # Accident
  0.15,   # Automatic
  0.20,   # Hybrid
  0.20    # Diesel
)

# ------------------------------------------------------------
# 9.3. Parametry priora dla wariancji błędu
# ------------------------------------------------------------

s_prior <- 0.6
s2_prior <- s_prior^2
v_prior <- 5

# ------------------------------------------------------------
# 9.4. Macierz U_prior
# ------------------------------------------------------------

U_prior <- diag((prior_sd^2) / s2_prior)

rownames(U_prior) <- colnames(X)
colnames(U_prior) <- colnames(X)

# ------------------------------------------------------------
# 9.5. Tabela zakresów a priori
# ------------------------------------------------------------

prior_ranges <- data.frame(
  Parameter = colnames(X),
  Prior_Mean = as.numeric(b_prior),
  Prior_SD = prior_sd,
  Prior_Lower_approx = as.numeric(b_prior) - 2 * prior_sd,
  Prior_Upper_approx = as.numeric(b_prior) + 2 * prior_sd
)

prior_ranges

# Kontrola
b_prior
U_prior
s2_prior
v_prior

stopifnot(nrow(b_prior) == k)
stopifnot(nrow(U_prior) == k)
stopifnot(ncol(U_prior) == k)

# ============================================================
# 10. Rozkład a posteriori w modelu normalnym-gamma
# ============================================================

# Macierze pomocnicze
U_prior_inv <- solve(U_prior)

# Parametry a posteriori
U_post <- solve(t(X) %*% X + U_prior_inv)

b_post <- U_post %*% (
  t(X) %*% y + U_prior_inv %*% b_prior
)

v_post <- v_prior + N

s2_post <- as.numeric(
  (v_prior * s2_prior +
     t(y) %*% y +
     t(b_prior) %*% U_prior_inv %*% b_prior -
     t(b_post) %*% solve(U_post) %*% b_post) / v_post
)

b_post
U_post
v_post
s2_post

# ============================================================
# 11. Tabela: wartości oczekiwane a priori i a posteriori
# ============================================================

posterior_summary <- data.frame(
  Parameter = colnames(X),
  Prior_Mean = as.numeric(b_prior),
  Posterior_Mean = as.numeric(b_post)
)

posterior_summary

# ============================================================
# 12. Rozkłady brzegowe a posteriori parametrów beta
# ============================================================

posterior_sd <- sqrt(s2_post * diag(U_post))

posterior_summary <- data.frame(
  Parameter = colnames(X),
  Prior_Mean = as.numeric(b_prior),
  Posterior_Mean = as.numeric(b_post),
  Posterior_SD = posterior_sd
)

posterior_summary

# ============================================================
# 13. Wykresy rozkładów brzegowych a posteriori
# ============================================================

# Folder na wykresy
if (!dir.exists("plots")) {
  dir.create("plots")
}

for (j in 1:k) {
  
  param_name <- colnames(X)[j]
  mean_j <- b_post[j, 1]
  scale_j <- posterior_sd[j]
  
  x_grid <- seq(
    mean_j - 4 * scale_j,
    mean_j + 4 * scale_j,
    length.out = 1000
  )
  
  density_j <- dt((x_grid - mean_j) / scale_j, df = v_post) / scale_j
  
  png(
    filename = paste0(
      "plots/posterior_density_",
      j, "_",
      gsub("[^A-Za-z0-9]", "", param_name),
      ".png"
    ),
    width = 900,
    height = 600
  )
  
  plot(
    x_grid, density_j,
    type = "l",
    lwd = 2,
    main = paste("Rozkład brzegowy a posteriori:", param_name),
    xlab = param_name,
    ylab = "Gęstość"
  )
  
  abline(v = mean_j, lty = 2, lwd = 2)
  
  dev.off()
}

posterior_summary
v_post
s2_post

# ============================================================
# 14. Przedziały HPDI 95% dla parametrów beta
# ============================================================

alpha <- 0.05

hpdi_lower <- as.numeric(b_post) + qt(alpha / 2, df = v_post) * posterior_sd
hpdi_upper <- as.numeric(b_post) + qt(1 - alpha / 2, df = v_post) * posterior_sd

posterior_summary$HPDI_95_Lower <- hpdi_lower
posterior_summary$HPDI_95_Upper <- hpdi_upper

# Czy HPDI zawiera zero?
posterior_summary$HPDI_contains_zero <- 
  posterior_summary$HPDI_95_Lower <= 0 &
  posterior_summary$HPDI_95_Upper >= 0

# Interpretacja znaków
posterior_summary$HPDI_interpretation <- ifelse(
  posterior_summary$HPDI_contains_zero,
  "HPDI zawiera 0 - efekt niejednoznaczny",
  ifelse(
    posterior_summary$HPDI_95_Lower > 0,
    "HPDI powyżej 0 - efekt dodatni",
    "HPDI poniżej 0 - efekt ujemny"
  )
)

posterior_summary

write.csv(posterior_summary, "posterior_summary_with_HPDI.csv", row.names = FALSE)

# ============================================================
# 15. Czynnik Bayesa
# ============================================================

# Funkcja licząca logarytm gęstości brzegowej p(y|M)
# w modelu normalnym-gamma.
# Jest to wzór z wykładu 5, liczony na logarytmach.

log_P_y_M <- function(y, X, Beta.prior, U.prior, s2.prior, v.prior) {
  
  N.data <- nrow(X)
  XTX <- t(X) %*% X
  
  Beta.ols <- solve(XTX) %*% t(X) %*% y
  
  v.data <- N.data - ncol(X)
  res <- y - X %*% Beta.ols
  s2.data <- as.numeric(t(res) %*% res / v.data)
  
  U.posterior <- solve(solve(U.prior) + XTX)
  
  v.posterior <- v.prior + N.data
  vs2.prior <- v.prior * s2.prior
  
  vs2.posterior <- v.prior * s2.prior +
    v.data * s2.data +
    t(Beta.ols - Beta.prior) %*%
    solve(U.prior + solve(XTX)) %*%
    (Beta.ols - Beta.prior)
  
  vs2.posterior <- as.numeric(vs2.posterior)
  
  # Logarytm wzoru z wykładu na p(y|M)
  wynik <- 
    0.5 * as.numeric(determinant(U.posterior, logarithm = TRUE)$modulus) +
    lgamma(v.posterior / 2) -
    (v.posterior / 2) * log(vs2.posterior) -
    (N.data / 2) * log(pi) -
    0.5 * as.numeric(determinant(U.prior, logarithm = TRUE)$modulus) -
    lgamma(v.prior / 2) +
    (v.prior / 2) * log(vs2.prior)
  
  return(wynik)
}

# ============================================================
# 16. Model pełny M1
# ============================================================

y_BF <- y
X_M1 <- X

Beta.prior_M1 <- b_prior
U.prior_M1 <- U_prior

P_y_M1_log <- log_P_y_M(
  y = y_BF,
  X = X_M1,
  Beta.prior = Beta.prior_M1,
  U.prior = U.prior_M1,
  s2.prior = s2_prior,
  v.prior = v_prior
)

P_y_M1_log

# ============================================================
# 17. Modele M2: model pełny bez jednej zmiennej
# ============================================================

zmienne <- c(
  "Age",
  "Milage_1000",
  "Engine_L",
  "Accident",
  "Automatic",
  "Hybrid",
  "Diesel"
)

P_y_M2_log <- rep(NA, length(zmienne))
names(P_y_M2_log) <- zmienne

for (i in 1:length(zmienne)) {
  
  usun <- zmienne[i]
  
  # Usuwamy daną kolumnę z X
  kolumna_do_usuniecia <- which(colnames(X_M1) == usun)
  
  X_M2 <- X_M1[, -kolumna_do_usuniecia]
  
  # Usuwamy odpowiadający parametr z priorów
  Beta.prior_M2 <- Beta.prior_M1[-kolumna_do_usuniecia, , drop = FALSE]
  
  U.prior_M2 <- U.prior_M1[
    -kolumna_do_usuniecia,
    -kolumna_do_usuniecia,
    drop = FALSE
  ]
  
  # Liczymy log p(y|M2)
  P_y_M2_log[i] <- log_P_y_M(
    y = y_BF,
    X = X_M2,
    Beta.prior = Beta.prior_M2,
    U.prior = U.prior_M2,
    s2.prior = s2_prior,
    v.prior = v_prior
  )
}

# ============================================================
# 18. Czynnik Bayesa BF(1,2)
# ============================================================

# BF(1,2) = P(y|M1) / P(y|M2)
# M1 = model pełny
# M2 = model bez danej zmiennej

log_BF_1_2 <- P_y_M1_log - P_y_M2_log

BF_1_2 <- exp(log_BF_1_2)

BF_table <- data.frame(
  zmienna = zmienne,
  log_BF = as.numeric(log_BF_1_2),
  log10_BF = as.numeric(log_BF_1_2 / log(10)),
  BF = as.numeric(BF_1_2)
)

BF_table$interpretacja <- ifelse(
  BF_table$BF < 1,
  "dane wspierają model bez tej zmiennej",
  ifelse(
    BF_table$BF < 3,
    "słabe wsparcie dla zmiennej",
    ifelse(
      BF_table$BF < 20,
      "umiarkowane wsparcie dla zmiennej",
      ifelse(
        BF_table$BF < 150,
        "silne wsparcie dla zmiennej",
        "bardzo silne wsparcie dla zmiennej"
      )
    )
  )
)

BF_table

write.csv(BF_table, "bayes_factors_results.csv", row.names = FALSE)

# ============================================================
# 19. Prognoza punktowa i przedziałowa dla wybranej obserwacji
# ============================================================

# Wybieramy jedną obserwację z próby
obs_id <- 1

selected_car <- data_model[obs_id, ]

selected_car[, c(
  "brand",
  "model",
  "model_year",
  "milage",
  "fuel_type",
  "engine",
  "transmission",
  "accident",
  "price",
  "Age",
  "Milage_1000",
  "Engine_L",
  "Accident",
  "Automatic",
  "Hybrid",
  "Diesel"
)]

# Wektor x_tau dla wybranej obserwacji
x_tau <- matrix(X[obs_id, ], nrow = 1)

# Prognoza punktowa w skali log ceny
forecast_log <- as.numeric(x_tau %*% b_post)

# Parametr skali rozkładu predykcyjnego
# y_tau | y ~ t(v_post, x_tau b_post,
#              sqrt(s2_post * (1 + x_tau U_post x_tau')))
forecast_scale_log <- sqrt(
  s2_post * as.numeric(1 + x_tau %*% U_post %*% t(x_tau))
)

# 95% przedział predykcyjny w skali log ceny
alpha <- 0.05

forecast_log_lower <- forecast_log +
  qt(alpha / 2, df = v_post) * forecast_scale_log

forecast_log_upper <- forecast_log +
  qt(1 - alpha / 2, df = v_post) * forecast_scale_log

# Przekształcenie z log ceny na cenę
forecast_price <- exp(forecast_log)
forecast_price_lower <- exp(forecast_log_lower)
forecast_price_upper <- exp(forecast_log_upper)

# Rzeczywista cena obserwacji
actual_price <- selected_car$price_num

prediction_result <- data.frame(
  brand = selected_car$brand,
  model = selected_car$model,
  model_year = selected_car$model_year,
  actual_price = actual_price,
  forecast_price = forecast_price,
  prediction_interval_95_lower = forecast_price_lower,
  prediction_interval_95_upper = forecast_price_upper
)

prediction_result

write.csv(prediction_result, "prediction_result.csv", row.names = FALSE)

selected_car[, c(
  "brand",
  "model",
  "model_year",
  "milage",
  "fuel_type",
  "engine",
  "transmission",
  "accident",
  "price"
)]

