functions {
	real lambda(real L, int m) {
		real lam;
		lam = ((m*pi())/(2*L))^2;
				
		return lam;
	}
	real spd(real alpha, real rho, real w) {
		real S;
		S = (alpha^2) * sqrt(2*pi()) * rho * exp(-0.5*(rho^2)*(w^2));
				
		return S;
	}
	vector phi(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
				
		return fi;
	}
}

data {
	real L;						//boundary condition factor
	int<lower=1> M;				//nº of basis functions		
	int<lower=1> N;				//nº total (training and test) observations
	int<lower=1> N1;			//nº training observations
	int ind1[N1];				//indices of training observations
	int<lower=1> N2;			//nº test observations
	int ind2[N2];				//indices of test observations
	vector[N] x;				//matrix of total (training and test) observations
	vector[N] y;
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
}

transformed parameters{
	vector[N1] f_train;
	vector[M] diagSPD;
	vector[M] SPD_beta;
	
	for(m in 1:M){ 
		diagSPD[m] =  sqrt(spd(alpha, rho, sqrt(lambda(L, m)))); 
	}
	
	SPD_beta = diagSPD .* beta;
	
	f_train= PHI[ind1,] * SPD_beta;
}

model{
	beta ~ normal(0,1);
	rho ~ normal(2,3);
	sigma ~ normal(0,1);
	alpha ~ normal(0,3);
	
	y[ind1] ~ normal(f_train, sigma); 
}

generated quantities{
	vector[N] f;
	vector[N] y_predict;
	vector[N] log_y_predict;	
	
	for(i in 1:N1){
		f[ind1[i]]= f_train[i];
	}
	for(i in 1:N2){
		f[ind2[i]]= PHI[ind2[i],] * SPD_beta;
	}
	
	for(i in 1:N){
		y_predict[i] = normal_rng(f[i], sigma);
		log_y_predict[i] = normal_lpdf(y[i] | f[i], sigma);
	}
}
