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
	int<lower=1> Nobs;					#nº uncensored observations 
	int<lower=1> obs[Nobs];				#uncensored observations
	int<lower=1> Ntrain;				#nº training observations (uncensored)
	int<lower=1> obs_train[Ntrain];		#training observations (uncensored)
	int<lower=0> Ntest;					#nº test observations (uncensored)
	int<lower=0> obs_test[Ntest];		#test observations (uncensored)
	
	int<lower=0> Ncens;					#nº censored observations
	int<lower=0> cens[Ncens];			#censored observations
	
	int<lower=1> D;						#nº dimensions
	vector[D] X[Nobs+Ncens+Ntest];		#matrix of inputs
	
	int<lower=0> Ngrid;					#nº grid inputs
	vector[D] Xgrid[Ngrid];				#matrix of grid inputs

	vector[Nobs+Ncens+Ntest] y;  		#observations
	
	int<lower=1> Nsex1;					#nº sex1 observations
	int<lower=1> sex1[Nsex1];			#sex1 observations
	int<lower=1> Nsex2;					#nº sex1 observations
	int<lower=1> sex2[Nsex2];			#sex1 observations
}

transformed data{

}

parameters {
	real<lower=0> rho[2]; 				#one lengthscale for each category (two categories: sex1 and sex2)
	
	real<lower=0> alpha[2];				# magnitud for each category (two categories: sex1 and sex2)
	
	real<lower=0> sigma;				# noise
	matrix[Nobs+Ncens+Ntest,2] eta;

}

transformed parameters{
	vector[Nobs+Ncens+Ntest] f;			# latent function for the base category (the base category is sex2)
	vector[Nsex1] f_sex1;				# latent function for category sex1 respect to base category
		
	# multilevel GP 3D
	f= gp123_one_lscale(X, alpha[1], rho[1], eta[,1]);
	f_sex1= gp123_one_lscale(X[sex1,], alpha[2], rho[2], eta[sex1,2]);
	f[sex1]= f[sex1] + f_sex1;
}

model{
	eta[,1] ~ normal(0,1);
	eta[,2] ~ normal(0,1);
	rho ~ normal(0,3);
	alpha ~ normal(0,5);
	sigma ~ normal(0,5);
	
	target += normal_lpdf(y[obs_train] | f[obs_train], sigma);
	target += normal_lccdf(y[cens] | f[cens], sigma); 
}

generated quantities{
	vector[Ngrid] f_predict_sex2;
	vector[Ngrid] f_predict_sex1;
	
	f_predict_sex2 = gp_posteriors_rng(Xgrid, f[sex2], X[sex2,], alpha[1], rho[1], 1e-12);
	f_predict_sex1 = gp_posteriors_rng(Xgrid, f_sex1, X[sex1,], alpha[2], rho[2], 1e-12);

}


