// generated with brms 2.8.0
functions {
}
data {
  int<lower=1> N;  // number of observations
  int Y[N];  // response variable
  int trials[N];  // number of trials
  // data for smooth terms
  int Ks;
  matrix[N, Ks] Xs;
  // data of smooth t2(x1,x2,x3,x4,k=2+kn[1])
  int nb_1;  // number of bases
  int knots_1[nb_1];
  matrix[N, knots_1[1]] Zs_1_1;
  matrix[N, knots_1[2]] Zs_1_2;
  matrix[N, knots_1[3]] Zs_1_3;
  matrix[N, knots_1[4]] Zs_1_4;
  matrix[N, knots_1[5]] Zs_1_5;
  matrix[N, knots_1[6]] Zs_1_6;
  matrix[N, knots_1[7]] Zs_1_7;
  matrix[N, knots_1[8]] Zs_1_8;
  matrix[N, knots_1[9]] Zs_1_9;
  matrix[N, knots_1[10]] Zs_1_10;
  matrix[N, knots_1[11]] Zs_1_11;
  matrix[N, knots_1[12]] Zs_1_12;
  matrix[N, knots_1[13]] Zs_1_13;
  matrix[N, knots_1[14]] Zs_1_14;
  matrix[N, knots_1[15]] Zs_1_15;
  int prior_only;  // should the likelihood be ignored?
  
  int<lower=1> N1;
  int ind1[N1];		# indices for training observations
  int<lower=1> N2;
  int ind2[N2];		# indices for testing observations
}
transformed data {
}
parameters {
  real temp_Intercept;  // temporary intercept
  // parameters for smooth terms
  vector[Ks] bs;
  // parameters of smooth t2(x1,x2,x3,x4,k=2+kn[1])
  vector[knots_1[1]] zs_1_1;
  real<lower=0> sds_1_1;
  vector[knots_1[2]] zs_1_2;
  real<lower=0> sds_1_2;
  vector[knots_1[3]] zs_1_3;
  real<lower=0> sds_1_3;
  vector[knots_1[4]] zs_1_4;
  real<lower=0> sds_1_4;
  vector[knots_1[5]] zs_1_5;
  real<lower=0> sds_1_5;
  vector[knots_1[6]] zs_1_6;
  real<lower=0> sds_1_6;
  vector[knots_1[7]] zs_1_7;
  real<lower=0> sds_1_7;
  vector[knots_1[8]] zs_1_8;
  real<lower=0> sds_1_8;
  vector[knots_1[9]] zs_1_9;
  real<lower=0> sds_1_9;
  vector[knots_1[10]] zs_1_10;
  real<lower=0> sds_1_10;
  vector[knots_1[11]] zs_1_11;
  real<lower=0> sds_1_11;
  vector[knots_1[12]] zs_1_12;
  real<lower=0> sds_1_12;
  vector[knots_1[13]] zs_1_13;
  real<lower=0> sds_1_13;
  vector[knots_1[14]] zs_1_14;
  real<lower=0> sds_1_14;
  vector[knots_1[15]] zs_1_15;
  real<lower=0> sds_1_15;
}
transformed parameters {
  vector[knots_1[1]] s_1_1 = sds_1_1 * zs_1_1;
  vector[knots_1[2]] s_1_2 = sds_1_2 * zs_1_2;
  vector[knots_1[3]] s_1_3 = sds_1_3 * zs_1_3;
  vector[knots_1[4]] s_1_4 = sds_1_4 * zs_1_4;
  vector[knots_1[5]] s_1_5 = sds_1_5 * zs_1_5;
  vector[knots_1[6]] s_1_6 = sds_1_6 * zs_1_6;
  vector[knots_1[7]] s_1_7 = sds_1_7 * zs_1_7;
  vector[knots_1[8]] s_1_8 = sds_1_8 * zs_1_8;
  vector[knots_1[9]] s_1_9 = sds_1_9 * zs_1_9;
  vector[knots_1[10]] s_1_10 = sds_1_10 * zs_1_10;
  vector[knots_1[11]] s_1_11 = sds_1_11 * zs_1_11;
  vector[knots_1[12]] s_1_12 = sds_1_12 * zs_1_12;
  vector[knots_1[13]] s_1_13 = sds_1_13 * zs_1_13;
  vector[knots_1[14]] s_1_14 = sds_1_14 * zs_1_14;
  vector[knots_1[15]] s_1_15 = sds_1_15 * zs_1_15;
}
model {
  vector[N1] mu_train = temp_Intercept + rep_vector(0, N1) + Xs[ind1,] * bs + Zs_1_1[ind1,] * s_1_1 + Zs_1_2[ind1,] * s_1_2 + Zs_1_3[ind1,] * s_1_3 + Zs_1_4[ind1,] * s_1_4 + Zs_1_5[ind1,] * s_1_5 + Zs_1_6[ind1,] * s_1_6 + Zs_1_7[ind1,] * s_1_7 + Zs_1_8[ind1,] * s_1_8 + Zs_1_9[ind1,] * s_1_9 + Zs_1_10[ind1,] * s_1_10 + Zs_1_11[ind1,] * s_1_11 + Zs_1_12[ind1,] * s_1_12 + Zs_1_13[ind1,] * s_1_13 + Zs_1_14[ind1,] * s_1_14 + Zs_1_15[ind1,] * s_1_15;
  // priors including all constants
  target += student_t_lpdf(temp_Intercept | 3, 0, 10);
  target += normal_lpdf(zs_1_1 | 0, 1);
  target += normal_lpdf(zs_1_2 | 0, 1);
  target += normal_lpdf(zs_1_3 | 0, 1);
  target += normal_lpdf(zs_1_4 | 0, 1);
  target += normal_lpdf(zs_1_5 | 0, 1);
  target += normal_lpdf(zs_1_6 | 0, 1);
  target += normal_lpdf(zs_1_7 | 0, 1);
  target += normal_lpdf(zs_1_8 | 0, 1);
  target += normal_lpdf(zs_1_9 | 0, 1);
  target += normal_lpdf(zs_1_10 | 0, 1);
  target += normal_lpdf(zs_1_11 | 0, 1);
  target += normal_lpdf(zs_1_12 | 0, 1);
  target += normal_lpdf(zs_1_13 | 0, 1);
  target += normal_lpdf(zs_1_14 | 0, 1);
  target += normal_lpdf(zs_1_15 | 0, 1);
  target += student_t_lpdf(sds_1_1 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_2 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_3 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_4 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_5 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_6 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_7 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_8 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_9 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_10 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_11 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_12 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_13 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_14 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sds_1_15 | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  // likelihood including all constants
  if (!prior_only) {
    target += binomial_logit_lpmf(Y[ind1] | trials[ind1], mu_train);
  }
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = temp_Intercept;
  
  	vector[N1+N2] mu;
	vector[N1+N2] y_predict;
	vector[N1+N2] f_invlogit;
	vector[N1+N2] log_y_predict;
	
	mu = temp_Intercept + rep_vector(0, N1) + Xs * bs + Zs_1_1 * s_1_1 + Zs_1_2 * s_1_2 + Zs_1_3 * s_1_3 + Zs_1_4 * s_1_4 + Zs_1_5 * s_1_5 + Zs_1_6 * s_1_6 + Zs_1_7 * s_1_7 + Zs_1_8 * s_1_8 + Zs_1_9 * s_1_9 + Zs_1_10 * s_1_10 + Zs_1_11 * s_1_11 + Zs_1_12 * s_1_12 + Zs_1_13 * s_1_13 + Zs_1_14 * s_1_14 + Zs_1_15 * s_1_15;
		
	for(i in 1:(N1+N2)){
		y_predict[i] = binomial_rng(1, inv_logit(mu[i]));
		f_invlogit[i] = inv_logit(mu[i]);

		log_y_predict[i] =  binomial_logit_lpmf(Y[i] | 1, mu[i]);
	}
}
