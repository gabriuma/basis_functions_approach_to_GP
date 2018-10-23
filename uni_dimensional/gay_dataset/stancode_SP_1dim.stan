// generated with brms 1.7.0
functions { 
} 

data { 
	int<lower=1> N;  				// total number of observations  
	int<lower=1> K;  				// number of population-level effects 
	matrix[N, K] X;  				// population-level design matrix 
	
	// vector[N] Y;  				// response variable(Normal model)
	int Y[N];						// response variable (Binomial model)
	int trials[N];  				// number of trials (Binomial model)
	 
	// data of smooth s(x)
	int nb_1;  						// number of basis 
	int knots_1[nb_1]; 
	matrix[N, knots_1[1]] Zs_1_1; 
	int prior_only;  				// should the likelihood be ignored? 

	int<lower=1> N1; 
	int ind1[N1];
} 

transformed data { 
	int Kc; 
	matrix[N, K - 1] Xc;  			// centered version of X 
	vector[K - 1] means_X; 			// column means of X before centering 
	Kc = K - 1;  					// the intercept is removed from the design matrix 
	for (i in 2:K) { 
		means_X[i - 1] = mean(X[, i]); 
		Xc[, i - 1] = X[, i] - means_X[i - 1]; 
	} 
} 

parameters { 
	vector[Kc] b;  					// population-level effects 
	real temp_Intercept;  			// temporary intercept 
	
	// parameters of smooth s(x)
	vector[knots_1[1]] zs_1_1; 
	real<lower=0> sds_1_1; 
} 

transformed parameters { 
	vector[knots_1[1]] s_1_1; 
	vector[N] mu; 
	
	s_1_1 = sds_1_1 * zs_1_1; 
	
	mu = Xc * b + Zs_1_1 * s_1_1 + temp_Intercept; 
} 

model { 
	// prior specifications 
	zs_1_1 ~ normal(0, 1); 
	sds_1_1 ~ student_t(3, 0, 10); 
	
	// likelihood contribution 
	if (!prior_only) { 
		Y[ind1] ~ binomial_logit(trials[ind1], mu[ind1]);
	} 
} 

generated quantities { 
	real b_Intercept;  				// population-level intercept 
	vector[N] y_predict;
	vector[N] log_y_predict;
	
	b_Intercept = temp_Intercept - dot_product(means_X, b);

	for(i in 1:N){
		y_predict[i] = binomial_rng(trials[i], inv_logit(mu[i]));
		log_y_predict[i] = binomial_logit_lpmf(Y[i] | trials[i], mu[i]);
	}
	
}
