
data {
	int<lower=1> N_pred;		# Number of observations	
	int<lower=1> N_train;		# Number of training observations
	
	vector[N_pred] x_pred;		# Input values
	vector[N_pred] y;			# Observations
	vector[N_pred] f_true;		# Generating function values		

	int vv_train[N_train];		# Training observation indices
	
	real L;						# Boundary
	int<lower=1> M;				# Number of basis functions
}

transformed data{
	matrix[N_train,M] PHI_train;
	matrix[N_pred,M] PHI_pred;

	for (m in 1:M){ 
		PHI_train[,m] = phi(L, m, x_pred[vv_train]);	# Eigenfunctions for training data
		PHI_pred[,m] = phi(L, m, x_pred); 				# Eigenfunctions for predicting data
	}
}

parameters {
	real<lower=0> lscale;		# Lengthscale hyperparameter
	real<lower=0> magnitude;	# Magnitud hyperparameter
	vector[M] beta;				# Weights
	real<lower=0> sigma;		# Noise
}

transformed parameters{
	vector[N_train] f;
	vector[M] diagSPD;
	vector[M] SPD_beta;

	for(m in 1:M){ 
		# Spectral densities
		diagSPD[m] =  sqrt(spd_Matern(magnitude, lscale, sqrt(lambda(L, m)))); 
	}
	
	# Latent HSGP function
	SPD_beta = diagSPD .* beta;
	f= PHI_train[,] * SPD_beta;
}

model{
	# Priors
	lscale ~ gamma(3.75, 25);
	sigma ~ gamma(1, 1);
	magnitude ~ gamma(5, 5);
	beta ~ normal(0, 1);
	
	# Likelihood
	y[vv_train] ~ normal(f, sigma); 
}

generated quantities{
	vector[N_pred] f_pred;
	
	# HSGP predictions
	f_pred= PHI_pred[,] * SPD_beta;
}

functions {
	# Eigenvalue
	real lambda(real L, int m) {
		real lam;
		lam = ((m*pi())/(2*L))^2;
				
		return lam;
	}
	
	# Spectral density of the Matern_3/2 kernel
	real spd_Matern(real magnitude, real lscale, real w, int v) {
		real S;

		if (v==3) {
			S = 4*magnitude^2 * (sqrt(3)/lscale)^3 * 1/((sqrt(3)/lscale)^2 + w^2)^2;
		}
		return S;
	}
	
	# Eigenfunction
	vector phi(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
				
		return fi;
	}
}