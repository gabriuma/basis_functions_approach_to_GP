// generated with brms 1.7.0
functions { 
} 
data { 
  int<lower=1> N;  // total number of observations 
  vector[N] Y;  // response variable 
  int<lower=1> K;  // number of population-level effects 
  matrix[N, K] X;  // population-level design matrix 
  // data of smooth s(x_sp,k=100)
  int nb_1;  // number of bases 
  int knots_1[nb_1]; 
  matrix[N, knots_1[1]] Zs_1_1; 
  int prior_only;  // should the likelihood be ignored? 
  
  int N_train;
  int vv_train[N_train];
  # vector[N] f_true;
} 
transformed data { 
  int Kc; 
  matrix[N, K - 1] Xc;  // centered version of X 
  vector[K - 1] means_X;  // column means of X before centering 
  Kc = K - 1;  // the intercept is removed from the design matrix 
  for (i in 2:K) { 
    means_X[i - 1] = mean(X[, i]); 
    Xc[, i - 1] = X[, i] - means_X[i - 1]; 
  } 
} 
parameters { 
  vector[Kc] b;  // population-level effects 
  real temp_Intercept;  // temporary intercept 
  // parameters of smooth s(x_sp,k=100)
  vector[knots_1[1]] zs_1_1; 
  real<lower=0> sds_1_1; 
  real<lower=0> sigma;  // residual SD 
} 
transformed parameters { 
  vector[knots_1[1]] s_1_1; 
  vector[N_train] mu; 
  
  s_1_1 = sds_1_1 * zs_1_1; 
  
  mu = Xc[vv_train] * b + Zs_1_1[vv_train,] * s_1_1 + temp_Intercept;
} 
model { 
 
  // prior specifications 
  zs_1_1 ~ normal(0, 1); 
  sds_1_1 ~ student_t(3, 0, 10); 
  sigma ~ student_t(3, 0, 10); 
  // likelihood contribution 
  if (!prior_only) { 
    Y[vv_train] ~ normal(mu, sigma); 
  } 
} 
generated quantities { 
  real b_Intercept;  // population-level intercept 
	vector[N] logdens_f;
	vector[N] logdens_y;
	
	vector[N] mu_pred;

  b_Intercept = temp_Intercept - dot_product(means_X, b);  
  
  mu_pred = Xc * b + Zs_1_1 * s_1_1 + temp_Intercept;
  
	for(i in 1:N){
		logdens_y[i] = normal_lpdf(Y[i] | mu_pred[i], sigma);
		# logdens_f[i] = normal_lpdf(f_true[i] | mu_pred[i], sigma);
	}

} 