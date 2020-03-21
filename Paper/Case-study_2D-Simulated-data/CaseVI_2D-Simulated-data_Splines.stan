// generated with brms 2.8.0
functions {
}
data {
  int<lower=1> N;  // number of observations
  vector[N] Y;  // response variable
  
	int<lower=1> Nsample;				# number of predicting/test observations
	int vv_sample[Nsample];
	
  // data for smooth terms
  int Ks;
  matrix[N, Ks] Xs;
  
  // data of smooth t2(x1,x2,k=3+kn[j],bs="tp")
  int nb_1;  // number of bases
  int knots_1[nb_1];
  matrix[N, knots_1[1]] Zs_1_1;
  matrix[N, knots_1[2]] Zs_1_2;
  matrix[N, knots_1[3]] Zs_1_3;
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
}
parameters {
  real temp_Intercept;  // temporary intercept
  
  // parameters for smooth terms
  vector[Ks] bs;
  
  // parameters of smooth t2(x1,x2,k=3+kn[j],bs="tp")
  vector[knots_1[1]] zs_1_1;
  real<lower=0> sds_1_1;
  vector[knots_1[2]] zs_1_2;
  real<lower=0> sds_1_2;
  vector[knots_1[3]] zs_1_3;
  real<lower=0> sds_1_3;
  real<lower=0> sigma;  // residual SD
}
transformed parameters {
  vector[knots_1[1]] s_1_1 = sds_1_1 * zs_1_1;
  vector[knots_1[2]] s_1_2 = sds_1_2 * zs_1_2;
  vector[knots_1[3]] s_1_3 = sds_1_3 * zs_1_3;
}
model {
  vector[Nsample] mu = temp_Intercept + rep_vector(0, Nsample) + Xs[vv_sample,] * bs + Zs_1_1[vv_sample,] * s_1_1 + Zs_1_2[vv_sample,] * s_1_2 + Zs_1_3[vv_sample,] * s_1_3;
  
  // priors including all constants
  target += student_t_lpdf(temp_Intercept | 3, 0, 10);
  target += normal_lpdf(zs_1_1 | 0, 1);
  target += normal_lpdf(zs_1_2 | 0, 1);
  target += normal_lpdf(zs_1_3 | 0, 1);
  target += student_t_lpdf(sds_1_1 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_2 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_3 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sigma | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
	
  // likelihood including all constants
  if (!prior_only) {
    target += normal_lpdf(Y[vv_sample] | mu, sigma);
  }
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = temp_Intercept;
  
	vector[N] mu_grid;
	vector[N] y_grid_predict;
	vector[N] log_y_predict;
	
	mu_grid = temp_Intercept + rep_vector(0, N) + Xs[,] * bs + Zs_1_1[,] * s_1_1 + Zs_1_2[,] * s_1_2 + Zs_1_3[,] * s_1_3;
	
	for(i in 1:(N)){
		y_grid_predict[i] = normal_rng(mu_grid[i], sigma);

		log_y_predict[i] = normal_lpdf(Y[i] | mu_grid[i], sigma);
	}
}
