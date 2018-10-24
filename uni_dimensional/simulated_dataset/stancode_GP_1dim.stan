functions {

	//Analytical posteriors
	vector gp_posteriors_rng(vector[] x2,
					 vector f1, vector[] x1,
					 real alpha, real rho, real delta) {
	int N1 = rows(f1);
	int N2 = size(x2);
	vector[N2] f2;
	{
	  matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho)
						 + diag_matrix(rep_vector(delta, N1));
	  matrix[N1, N1] L_K = cholesky_decompose(K);

	  vector[N1] L_K_div_f1 = mdivide_left_tri_low(L_K, f1);
	  vector[N1] K_div_f1 = mdivide_right_tri_low(L_K_div_f1', L_K)';
	  matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, alpha, rho);
	  vector[N2] f2_mu = (k_x1_x2' * K_div_f1);
	  matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
	  matrix[N2, N2] cov_f2 =   cov_exp_quad(x2, alpha, rho) - v_pred' * v_pred
							  + diag_matrix(rep_vector(delta, N2));
	  f2 = multi_normal_rng(f2_mu, cov_f2);
	}
	return f2;
	}
	
	//GP covariance function
	vector gp(vector[] x, real sdgp, real lscale, vector zgp) { 
		matrix[size(x), size(x)] cov;
		cov = cov_exp_quad(x, sdgp, lscale);
		for (n in 1:size(x)) {
			cov[n, n] = cov[n, n] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
}

data {
	int<lower=1> N1;			//nº training observations
	int ind1[N1];				//indices of training observations
	int<lower=0> N2;			//nº test observations
	int ind2[N2];				//indices of test observations
	int<lower=1> N;				//nº total (training and test) observations
	vector[1] x[N];				//matrix of total (training and test) observations
	vector[N] y;				//response variable (for normal model)	
}

transformed data{
}

parameters {
	real<lower=0> rho;
	real<lower=0> sigma;
	real<lower=0> alpha;
	vector[N1] eta;
}

transformed parameters{
	vector[N1] f_train;
	
	f_train= gp(x[ind1], alpha, rho, eta);
}

model{
	eta ~ normal(0, 1);
	rho ~ normal(0, 5);
	sigma ~ normal(0, 1);
	alpha ~ normal(0, 3);
	
	y[ind1] ~ normal(f_train, sigma); 
}

generated quantities{
	vector[N] f;
	vector[N] y_predict;
	vector[N] log_y_predict;	
	
	f[ind2] = gp_posteriors_rng(x[ind2], f_train, x[ind1], alpha, rho, 1e-12);
	
	for(i in 1:N1){
		f[ind1[i]]= f_train[i];
	}

	for(i in 1:N){
		y_predict[i] = normal_rng(f[i], sigma);
		log_y_predict[i] = normal_lpdf(y[i] | f[i], sigma);
	}
}
