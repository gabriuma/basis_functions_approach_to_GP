functions {
	real lambda(real L, int m) {
		real lam;
		lam = ((m*pi())/(2*L))^2;
				
		return lam;
	}
	real spd(real alpha, real rho, real w) {
		real S;
		S = (alpha^2)*sqrt(2*pi())*rho*exp(-0.5*(rho^2)*(w^2));
				
		return S;
	}
	vector phi(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L)*sin((m*pi())/(2*L) * (x+L));
				
		return fi;
	}
}

data {
	real L;						#boundary condition' factor
	int<lower=1> M;				#nº of basis functions		
	int<lower=1> N;				#nº total (training and test) observations
	int<lower=1> N1;			#nº training observations
	int ind1[N1];				#indices of training observations
	vector[N] x;				#matrix of total (training and test) observations
	vector[N] y;
	# int n[N];
	# int y_int[N];
}

transformed data {
	matrix[N,M] PHI;
	for (m in 1:M){ PHI[,m] = phi(L, m, x); }
}

parameters {
	vector[M] beta;
	real<lower=0> rho;
	real<lower=0> sigma;
	real<lower=0> alpha;
	real c0;
	real c1;
}

transformed parameters{
 
	vector[N] f;
	
	# BASIS FUNCTIONS (version 1)
	# matrix[M,M] SPD;
	
	# for (i in 1:M){ for (j in 1:M){ SPD[i,j] = 0; }	}
	# for (m in 1:M){ SPD[m,m] = sqrt(spd(alpha, rho, sqrt(lambda(L, m)))); }

	# f= c0 + x*c1 + PHI * SPD * beta;
	# f= PHI * SPD * beta;
	
	# BASIS FUNCTIONS (version 2)
	vector[M] diagSPD;
	vector[M] SPD_beta;
	
	for(m in 1:M){ diagSPD[m] =  sqrt(spd(alpha, rho, sqrt(lambda(L, m)))); }
	
	SPD_beta = diagSPD .* beta;
	
	f= c0 + x*c1 + PHI * SPD_beta;
	# f= PHI * SPD_beta;
   
}

model{
	beta ~ normal(0,1);
	rho ~ normal(0,1);
	sigma ~ normal(0,1);
	alpha ~ normal(0,1);
	c0 ~ normal(0,1);
	c1 ~ normal(0,1);
	
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


