functions {
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
	int<lower=1> N1;			#nº training observations
	int ind1[N1];				#indices of training observations
	int<lower=1> N;				#nº total (training and test) observations
	vector[1] x[N];				#matrix of total (training and test) observations
	
	vector[N] y;				#response variable (for normal model)
	
	# int n[N];					#trials of binomial observations (for binomial model)
	# int y_int[N];				#binomial response variable (for binomial model)	
}

transformed data{
	vector[N] x_1; 				#product in linear component

	for(i in 1:N){
		x_1[i] = x[i][1]; }
}

parameters {
	real<lower=0> rho;
	real<lower=0> sigma;
	real<lower=0> alpha;
	vector[N] eta;
	# real c0;					#product in linear component
	# real c1;
}

transformed parameters{
	vector[N] f;
	
	# f= c0 + x_1*c1 + gp(x, alpha, rho, eta); 
	f= gp(x, alpha, rho, eta);
}

model{
	eta ~ normal(0, 1);
	rho ~ normal(2, 3);
	sigma ~ normal(0, 1);
	alpha ~ normal(0, 5);
	# c0 ~ normal(0,1);			#product in linear component
	# c1 ~ normal(0,1);
	
	y[ind1] ~ normal(f[ind1], sigma); 
	# y_int[ind1] ~ binomial_logit(n[ind1], f[ind1]); 
}

generated quantities{
	vector[N] y_predict;
	vector[N] log_y_predict;	

	for(i in 1:N){
		y_predict[i] = normal_rng(f[i], sigma);
		# y_predict[i] = binomial_rng(n[i], inv_logit(f[i]));

		log_y_predict[i] = normal_lpdf(y[i] | f[i], sigma);
		# log_y_predict[i] = binomial_logit_lpmf(y_int[i] | n[i], f[i]);
	}

}


