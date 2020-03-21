
data {
	int<lower=1> N;				# Total number (training and test) of observations
	int<lower=1> N1;			# Number of training observations
	int<lower=1> N2;			# Number of test observations
	
	vector[N] x;				# Matrix of inputs for training and test observations
	int n[N];					# Trials of binomial observations
	int y[N];					# Binomial response variable
	
	int ind1[N1];				# Training observation indices
	int ind2[N2];				# Test observation indices

	real L;						# Boundary	
	int<lower=1> M;				# Number of basis functions
}

transformed data {
	matrix[N,M] PHI;
	for (m in 1:M){ PHI[,m] = phi(L, m, x); }		# Eigenfunctions for training  and test data
}

parameters {
	real<lower=0> lscale;		# Lengthscale hyperparameter
	real<lower=0> magnitud;		# Magnitud hyperparameter
	vector[M] beta;				# Weights
	real c0;					# Linear coeficient (intercept)
	real c1;					# Linear coeficient
}

transformed parameters{
	vector[N1] f_train;
	vector[M] diagSPD;
	vector[M] SPD_beta;
	
	for(m in 1:M){ 
		# Spectral densities
		diagSPD[m] =  sqrt(spd(magnitud, lscale, sqrt(lambda(L, m)))); 
	}
	
	# Linear model + latent HSGP function
	SPD_beta = diagSPD .* beta;
	f_train=  c0 + x[ind1]*c1 + PHI[ind1,] * SPD_beta;  
}

model{
	# Priors
	lscale ~ inv_gamma(2,1);
	magnitud ~ normal(0,1);
	beta ~ normal(0,1);
	c0 ~ normal(0,1);
	c1 ~ normal(0,1);

	# Likelihood	 
	y[ind1] ~ binomial_logit(n[ind1], f_train);
}

generated quantities{
	vector[N] f;
	vector[N] y_predict;
	vector[N] log_y_predict;	
	
	for(i in 1:N1){
		# Linear function values
		f[ind1[i]]= f_train[i];
	}
	for(i in 1:N2){
		# Linear and HSGP function predictions
		f[ind2[i]]= c0 + x[ind2[i]]*c1 + PHI[ind2[i],] * SPD_beta; 
	}
	
	for(i in 1:N){
		# Predicting observations
		y_predict[i] = binomial_rng(n[i], inv_logit(f[i]));
		
		# Log predictive density
		log_y_predict[i] = binomial_logit_lpmf(y[i] | n[i], f[i]);
	}
}

functions {
	# Eigenvalue
	real lambda(real L, int m) {
		real lam;
		lam = (m*pi()/(2*L))^2;
				
		return lam;
	}
	
	# Spectral density of the squared exponential kernel
	real spd(real magnitud, real lscale, real w) {
		real S;
		S = (magnitud^2) * sqrt(2*pi()) * lscale * exp(-0.5*(lscale^2)*(w^2));
				
		return S;
	}
	
	# Eigenfunction
	vector phi(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
				
		return fi;
	}
}
