functions {
	real lambda(real L, int m) {
		real lam;
		lam = ((m*pi())/(2*L))^2;
				
		return lam;
	}
	real spd(real alpha, real rho, real w) {
		real s;
		s = (alpha^2)*sqrt(2*pi()*rho^2)*exp(-0.5*(rho^2)*(w^2));
				
		return s;
	}
	vector phi(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
				
		return fi;
	}
	matrix PHI(real L, vector x, int M) {
		matrix[rows(x),M] FI;
		for (m in 1:M){ FI[,m] = phi(L, m, x); }
				
		return FI;
	}
	vector SPD(real L, real alpha, real rho, int M) {
		# matrix[M,M] S;
		# for (i in 1:M){ for (j in 1:M){ S[i,j] = 0; }	}
		# for (m in 1:M){ S[m,m] = sqrt(spd(alpha, rho, sqrt(lambda(L, m)))); }
		
		vector[M] diagS;
		for(m in 1:(M)){ diagS[m] = sqrt(spd(alpha, rho, sqrt(lambda(L, m)))); }
				
		return diagS;
	}
	vector SPD_beta(real L, real alpha, real rho, vector beta, int M) {
		vector[M] diagS_beta;
		diagS_beta = SPD(L, alpha, rho, M) .* beta;
				
		return diagS_beta;
	}
	vector f(real L, real alpha, real rho, vector x, vector beta, int M) {
		# vector[rows(x)] fx;
		# fx= PHI(L, x, M) * SPD(L, alpha, rho, M) * beta;
		
		vector[rows(x)] fx;
		fx= PHI(L, x, M) * SPD_beta(L, alpha, rho, beta, M);
				
		return fx;
	}
}

data {
	int<lower=1> D;				# nº dimensions
	real L[D];					# boundary condition' factor
	int<lower=1> M;				# nº basis function
	int<lower=1> N1;			# nº training observations
	int Npred;					# nº predicting/test observations
	matrix[N1+Npred,D] x;		# matrix of total training and predicting/test inputs
	vector[N1+Npred] y;			# vector of observations
}

transformed data {

}

parameters {
	matrix[M,D] beta;
	real<lower=0> rho[D];
	real<lower=0> sigma;
	real<lower=0> alpha[D];
}

transformed parameters{
	vector[N1+Npred] F;
	F= f(L[1], alpha[1], rho[1], x[,1], beta[,1], M) + f(L[2], alpha[2], rho[2], x[,2], beta[,2], M);
}

model{
	beta[,1] ~ normal(0,1);
	beta[,2] ~ normal(0,1);
	rho ~ normal(0,2);
	sigma ~ normal(0,1);
	alpha ~ normal(0,3);
	
	y[1:N1] ~ normal(F[1:N1], sigma); 
}

generated quantities{
	vector[N1+Npred] y_predict;
	vector[N1+Npred] log_y_predict;

	for(i in 1:(N1+Npred)){
		y_predict[i] = normal_rng(F[i], sigma);

		log_y_predict[i] = normal_lpdf(y[i] | F[i], sigma);
	}
	
}


