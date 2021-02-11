
data {
	int<lower=1> N_train;		# Number of training observations		
	int<lower=1> N_pred;		# Number of observations	
	
	real x_pred[N_pred];		# Input values	
	vector[N_pred] y_pred;		# Observations	

	int vv_train[N_train];		# Training observation indices  	
}

transformed data{
}

parameters {
	real<lower=0> lscale;		# Lengthscale hyperparameter
	real<lower=0> magnitud;		# Magnitud hyperparameter
	vector[N_train] eta;		# Standard normal variable
	real<lower=0> sigma;		# Noise
}

transformed parameters{
	vector[N_train] f;

	# GP latent function
	f = gp_mattern(x_pred[vv_train], magnitud, lscale, eta);
}

model{
	# Priors
	eta ~ normal(0, 1);
	lscale ~ gamma(3.75, 25);
	sigma ~ gamma(1, 1);
	magnitud ~ gamma(5, 5);
	
	# Likelihood
	y_pred[vv_train] ~ normal(f, sigma); 
}

generated quantities{
	vector[N_pred] f_pred;
	
	# Analytic predictions of GP function values
	f_pred = gp_matern_marginal(x_pred, f, x_pred[vv_train], magnitud, lscale, 1e-12);
}

functions {
	# GP with Mattern_3/2 kernel
	vector gp_mattern(real[] x, real sdgp, real lscale, vector zgp) {
		int n= size(x);	
		matrix[n, n] cov;

		for (i in 1:n) {
			for (j in 1:n) {
				cov[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x[i]-x[j])))/lscale) * exp(-sqrt(3)*(fabs(x[i]-x[j]))/lscale);
			}
		}
		
		for (i in 1:n) {
			cov[i, i] = cov[i, i] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
	
	# Marginal GP with Mattern kernel
	vector gp_matern_marginal(real[] x2, vector f1, real[] x1, real magnitud, real lscale, real delta) {
	int N1 = rows(f1);
	int N2 = size(x2);
	vector[N2] f2;
	matrix[N1, N1] K;
	matrix[N1, N2] k_x1_x2;
	matrix[N2, N2] cov_f2;
	matrix[N1, N1] L_K;
	vector[N1] L_K_div_f1;
	vector[N1] K_div_f1;
	vector[N2] f2_mu;
	matrix[N1, N2] v_pred;
	{
	  for (i in 1:N1) {
		for (j in i:N1) {
			K[i,j] = magnitud^2 * (1 + (sqrt(3)*(fabs(x1[i]-x1[j])))/lscale) * exp(-sqrt(3)*(fabs(x1[i]-x1[j]))/lscale);
		}
	  }
	  for (i in 1:N1) {
	     K[i,i] = K[i, i] + delta;
	     for (j in (i+1):N1) {
			K[j, i] = K[i, j];
		 }
	  }
	  
	  L_K = cholesky_decompose(K);
	  L_K_div_f1 = mdivide_left_tri_low(L_K, f1);
	  K_div_f1 = mdivide_right_tri_low(L_K_div_f1', L_K)';

	  for (i in 1:N1) {
		for (j in 1:N2) {
			k_x1_x2[i,j] = magnitud^2 * (1 + (sqrt(3)*(fabs(x1[i]-x2[j])))/lscale) * exp(-sqrt(3)*(fabs(x1[i]-x2[j]))/lscale);
		}
	  }
	  
	  f2_mu = (k_x1_x2' * K_div_f1); //'
	  v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
	  
	  for (i in 1:N2) {
		for (j in i:N2) {
			cov_f2[i,j] = magnitud^2 * (1 + (sqrt(3)*(fabs(x2[i]-x2[j])))/lscale) * exp(-sqrt(3)*(fabs(x2[i]-x2[j]))/lscale);
		}
	  }
	  for (i in 1:N2) {
	     for (j in (i+1):N2) {
			cov_f2[j, i] = cov_f2[i, j];
		 }
	  }	  
	  
	  cov_f2 =   cov_f2 - v_pred' * v_pred + diag_matrix(rep_vector(delta, N2)); //'
	  f2 = multi_normal_rng(f2_mu, cov_f2);
	}
	return f2;
	}
}
