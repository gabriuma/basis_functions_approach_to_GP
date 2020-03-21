functions {
	#GP 1D
	vector gp(real[] x, real sdgp, real lscale, vector zgp) { 
		matrix[size(x), size(x)] cov;
		cov = cov_exp_quad(x, sdgp, lscale);
		for (n in 1:size(x)) {
			cov[n, n] = cov[n, n] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}

	#Additive-2D GPs
	vector gp1gp2(real[] x1, real[] x2, real sdgp1, real sdgp2, real lscale1, real lscale2, vector zgp) { 
		matrix[size(x1), size(x1)] cov;
		cov = cov_exp_quad(x1, sdgp1, lscale1) + cov_exp_quad(x2, sdgp2, lscale2);
		for (n in 1:size(x1)) {
			cov[n, n] = cov[n, n] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
	
	#GP 2D
	vector gp12(real[] x1, real[] x2, real sdgp, real lscale1, real lscale2, vector zgp) { 
		matrix[size(x1), size(x1)] cov;
		cov = cov_exp_quad(x1, sdgp, lscale1).*cov_exp_quad(x2, 1, lscale2);
		for (n in 1:size(x1)) {
			cov[n, n] = cov[n, n] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
	#GP 2D just one lengthscale
	vector gp12_one_lscale(vector[] x, real sdgp, real lscale, vector zgp) { 
		matrix[size(x), size(x)] cov;
		
		cov = cov_exp_quad(x, sdgp, lscale);
	
		for (i in 1:size(x)) {
			cov[i, i] = cov[i, i] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
	
	#GP 3D
	vector gp123(real[] x1, real[] x2, real[] x3, real sdgp, real lscale1, real lscale2, real lscale3, vector zgp) { 
		matrix[size(x1), size(x1)] cov;
		
		cov = cov_exp_quad(x1, sdgp, lscale1).*cov_exp_quad(x2, 1, lscale2).*cov_exp_quad(x3, 1, lscale3);
		
		#alternative cov
		# for (i in 1:(size(x1)-1)){
			# for (j in (i+1):size(x1)){
				# cov[i,j] = square(sdgp) * exp(-0.5 
							# * ( 1/square(lscale1) * square(x1[i]-x1[j]) 
							# + 1/square(lscale2) * square(x2[i]-x2[j]) 
							# + 1/square(lscale3) * square(x3[i]-x3[j]) ) );
							
				# cov[j,i] = cov[i,j];
			# }
		# }

		for (i in 1:size(x1)) {
			cov[i, i] = cov[i, i] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
	
	#GP 3D just one lengthscale
	vector gp123_one_lscale(vector[] x, real sdgp, real lscale, vector zgp) { 
		matrix[size(x), size(x)] cov;
		
		cov = cov_exp_quad(x, sdgp, lscale);
	
		for (i in 1:size(x)) {
			cov[i, i] = cov[i, i] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
	#Analytical posteriors
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
}

data {
	int<lower=1> N1;				#nº training observations 
	int<lower=1> N2;				#nº test observations
	int ind1[N1];					#indices for training observations
	int ind2[N2];					#indices for test observations
	
	int<lower=0> Npred;				#nº grid-predicting input observations
	
	int<lower=1> D;					#nº dimensions
	vector[D] X[N1+N2+Npred];		#matrix of inputs
	# int ind_trai_test[N1+N2];
	# int indpred[Npred];
	
	int y[N1+N2];  					#observations (binomial observations)
	int trials[N1+N2];  			#trials for each binomial observation	
}

transformed data{
	vector[N1+N2+Npred] x_1; 		#for product in linear component
	vector[N1+N2+Npred] x_2;

	for(i in 1:(N1+N2+Npred)){
		x_1[i] = X[i][1]; 
		x_2[i] = X[i][2]; 
	}
}

parameters {
	real<lower=0> rho[1];
	real<lower=0> alpha[1];
	matrix[N1,1] eta;
	# real c0;						#coefficients in linear components
	# real c1;
	# real c2;
}

transformed parameters{
	vector[N1] f_train;

	#GP 2D
	# f= gp12(X[,1], X[,2], alpha[1], rho[1], rho[2], eta[,1]);
	f_train= gp12_one_lscale(X[ind1], alpha[1], rho[1], eta[,1]);
	
	#Additive GPs
	# f= gp1gp2(X[,1], X[,2], alpha[1], alpha[2], rho[1], rho[2], eta[,1]);
	
	#GP 3D
	# f= gp123(X[,1], X[,2], X[,3], alpha[1], rho[1], rho[2], rho[3], eta[,1]);
	# f= gp123_one_lscale(X[ind_trai_test], alpha[1], rho[1], eta[,1]);
}

model{
	eta[,1] ~ normal(0,1);
	# eta[,2] ~ normal(0,1);
	rho ~ gamma(4,1);
	alpha ~ inv_gamma(2,5);
	# c0 ~ normal(0,1);				#coefficients in linear components
	# c1 ~ normal(0,1);
	# c2 ~ normal(0,1);
	
	y[ind1] ~ binomial_logit(trials[ind1], f_train);   
}

generated quantities{
	vector[N1+N2] f;
	vector[N2] f2;
	vector[N1+N2] y_predict;
	vector[N1+N2] f_invlogit;	
	vector[N1+N2] log_y_predict;
	
	f2 = gp_posteriors_rng(X[ind2], f_train, X[ind1], alpha[1], rho[1], 1e-12);
	
	for(i in 1:(N1)){
		f[ind1[i]] = f_train[i];
	}
	for(i in 1:(N2)){
		f[ind2[i]] = f2[i];
	}

	for(i in 1:(N1+N2)){
		y_predict[i] = binomial_rng(1, inv_logit(f[i]));	
		f_invlogit[i] = inv_logit(f[i]);  #bernoulli_logit_rng(f[i])
	}
	
	for(i in 1:(N1+N2)){
		log_y_predict[i] = binomial_logit_lpmf(y[i] | trials[i], f[i]);
	}

}


