
data {
	int<lower=1> N1;			# Number of training observations
	int<lower=0> N2;			# Number of test observations
	int<lower=1> N;				# Total number (training and test) of observations
	
	vector[1] x[N];				# Matrix of inputs for training and test observations
	int n[N];					# Trials of binomial observations
	int y[N];					# Binomial response variable	

	int ind1[N1];				# Training observation indices
	int ind2[N2];				# Testing observation indices
}

transformed data{
	vector[N] x_1; 				

	for(i in 1:N){
		x_1[i] = x[i][1]; }		# For the product in the linear component
}

parameters {
	real<lower=0> lscale;		# Lengthscale hyperparameter
	real<lower=0> magnitud;		# Magnitud hyperparameter
	vector[N1] eta;				# Standard normal variable
	real c0;					# Linear coeficient (intercept)
	real c1;					# Linear coeficient
}

transformed parameters{
	vector[N1] f_train;
	vector[N1] f_train_linear;

	# GP latent function
	f_train= gp(x[ind1], magnitud, lscale, eta); 
	
	# Linear model
	f_train_linear= c0 + x_1[ind1]*c1 + f_train;
}

model{
	# Priors
	eta ~ normal(0, 1);
	lscale ~ inv_gamma(2,1);
	magnitud ~ normal(0, 5);
	c0 ~ normal(0,1);
	c1 ~ normal(0,1);
	
	# Likelihood
	y[ind1] ~ binomial_logit(n[ind1], f_train_linear); 
}

generated quantities{
	vector[N2] f_test;
	vector[N] f;
	vector[N] y_predict;
	vector[N] log_y_predict;
	
	for(i in 1:N1){
		f[ind1[i]]= f_train_linear[i];		# Linear function values
	}
	
	# Analytic predictions of GP function values
	f_test = gp_posteriors_marginal(x[ind2], f_train, x[ind1], magnitud, lscale, 1e-12);
	
	
	for(i in 1:N2){
		f[ind2[i]] = c0 + x_1[ind2[i]]*c1 + f_test[i];		# Linear model
	}
	
	for(i in 1:N){
		# Predicting observations
		y_predict[i] = binomial_rng(n[i], inv_logit(f[i]));
		
		# Log predictive density
		log_y_predict[i] = binomial_logit_lpmf(y[i] | n[i], f[i]);
	}
}

functions {
	# GP with squared exponential kernel
	vector gp(vector[] x, real sdgp, real lscale, vector zgp) { 
		matrix[size(x), size(x)] cov;
		cov = cov_exp_quad(x, sdgp, lscale);
		for (n in 1:size(x)) {
			cov[n, n] = cov[n, n] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}

	# Marginal GP with squared exponential kernel
	vector gp_posteriors_marginal(vector[] x2,
					 vector f1, vector[] x1,
					 real magnitud, real lscale, real delta) { 
	int N1 = rows(f1);
	int N2 = size(x2);
	vector[N2] f2;
	{
	  matrix[N1, N1] K =   cov_exp_quad(x1, magnitud, lscale)
						 + diag_matrix(rep_vector(delta, N1));
	  matrix[N1, N1] L_K = cholesky_decompose(K);

	  vector[N1] L_K_div_f1 = mdivide_left_tri_low(L_K, f1);
	  vector[N1] K_div_f1 = mdivide_right_tri_low(L_K_div_f1', L_K)';
	  matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, magnitud, lscale);
	  vector[N2] f2_mu = (k_x1_x2' * K_div_f1);
	  matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
	  matrix[N2, N2] cov_f2 =   cov_exp_quad(x2, magnitud, lscale) - v_pred' * v_pred
							  + diag_matrix(rep_vector(delta, N2));
	  f2 = multi_normal_rng(f2_mu, cov_f2);
	}
	return f2;
	}
}
