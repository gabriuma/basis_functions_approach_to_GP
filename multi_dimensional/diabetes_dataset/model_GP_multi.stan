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
}

data {
	int<lower=1> N1;				#nº training observations 
	int<lower=1> N2;				#nº test observations
	int ind1[N1];					#indices for training observations
	int ind2[N2];					#indices for test observations
	
	int<lower=0> Npred;				#nº grid-predicting input observations
	
	int<lower=1> D;					#nº dimensions
	vector[D] X[N1+N2+Npred];		#matrix of inputs
	
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
	real<lower=0> rho[D];
	real<lower=0> alpha[D];
	matrix[N1+N2+Npred,D] eta;
	# real c0;						#coefficients in linear components
	# real c1;
	# real c2;
}

transformed parameters{
	vector[N1+N2+Npred] f;

	#GP 2D
	# f= gp12(X[,1], X[,2], alpha[1], rho[1], rho[2], eta[,1]);
	# f= gp12_one_lscale(X, alpha[1], rho[1], eta[,1]);
	
	#Additive GPs
	# f= gp1gp2(X[,1], X[,2], alpha[1], alpha[2], rho[1], rho[2], eta[,1]);
	
	#GP 3D
	# f= gp123(X[,1], X[,2], X[,3], alpha[1], rho[1], rho[2], rho[3], eta[,1]);
	f= gp123_one_lscale(X, alpha[1], rho[1], eta[,1]);
}

model{
	eta[,1] ~ normal(0,1);
	eta[,2] ~ normal(0,1);
	rho ~ normal(3,2);
	alpha ~ normal(0,4);
	# c0 ~ normal(0,1);				#coefficients in linear components
	# c1 ~ normal(0,1);
	# c2 ~ normal(0,1);
	
	y[ind1] ~ binomial_logit(trials[ind1], f[ind1]);   
}

generated quantities{
	vector[N1+N2+Npred] y_predict;
	vector[N1+N2+Npred] f_invlogit;	
	vector[N2] log_y_predict;

	for(i in 1:(N1+N2+Npred)){
		y_predict[i] = binomial_rng(1, inv_logit(f[i]));	
		f_invlogit[i] = inv_logit(f[i]);  #bernoulli_logit_rng(f[i])
	}
	
	for(i in 1:N2){
		log_y_predict[i] = binomial_logit_lpmf(y[ind2[i]] | trials[ind2[i]], f[ind2[i]]);
	}

}


