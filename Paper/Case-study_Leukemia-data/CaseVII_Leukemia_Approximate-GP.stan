functions {
	real lambda(real L, int m) {
		real lam;
		lam = ((m*pi())/(2*L))^2;
				
		return lam;
	}
	vector lambda_nD(real[] L, int[] m, int D) {
		vector[D] lam;
		for(i in 1:D){
			lam[i] = ((m[i]*pi())/(2*L[i]))^2; }
				
		return lam;
	}
	real spd(real alpha, real rho, real w) {
		real S;
		S = (alpha^2) * sqrt(2*pi()) * rho * exp(-0.5*(rho^2)*(w^2));
				
		return S;
	}
	real spd_2D(real alpha, real rho1, real rho2, real w1, real w2) {
		real S;
		S = alpha^2 * sqrt(2*pi())^2 * rho1*rho2 * exp(-0.5*(rho1^2*w1^2 + rho2^2*w2^2));
				
		return S;
	}
	real spd_nD(real alpha, row_vector rho, vector w, int D) {
		real S;
		S = alpha^2 * sqrt(2*pi())^D * prod(rho) * exp(-0.5*((rho .* rho) * (w .* w)));
				
		return S;
	}
	real spd_nD_onelscale(real alpha, real rho, vector w, int D) {
		real S;
		S = alpha^2 * sqrt(2*pi())^D * rho^D * exp(-0.5*rho^2 * (w' * w)); #'
				
		return S;
	}
	vector phi(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
				
		return fi;
	}
	vector phi_2D(real L1, real L2, int m1, int m2, vector x1, vector x2) {
		vector[rows(x1)] fi;
		vector[rows(x1)] fi1;
		vector[rows(x1)] fi2;
		fi1 = 1/sqrt(L1)*sin(m1*pi()*(x1+L1)/(2*L1));
		fi2 = 1/sqrt(L2)*sin(m2*pi()*(x2+L2)/(2*L2));
		fi = fi1 .* fi2;
		return fi;
	}
	vector phi_nD(real[] L, int[] m, matrix x) {
		int c = cols(x);
		int r = rows(x);
		
		matrix[r,c] fi;
		vector[r] fi1;
		for (i in 1:c){
			fi[,i] = 1/sqrt(L[i])*sin(m[i]*pi()*(x[,i]+L[i])/(2*L[i]));
		}
		fi1 = fi[,1];
		for (i in 2:c){
			fi1 = fi1 .* fi[,i];
		}
		return fi1;
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
	
	int<lower=0> Nx1grid;				#nº grid inputs
	
	int<lower=1> D;						#nº dimensions
	matrix[Nobs+Ncens,D] X;				#matrix of inputs
	matrix[Nx1grid,D] X1;				#matrix of inputs
	matrix[Nx1grid,D] X2;				#matrix of inputs
	matrix[Nx1grid,D] X3;				#matrix of inputs
	matrix[Nx1grid,D] X2_1;				#matrix of inputs
	matrix[Nx1grid,D] X2_2;				#matrix of inputs
	
	vector[Nobs+Ncens] y;  				#observations

	real L[D];							#boundary condition' factor
	int<lower=1> M;						#nº basis functions
	int<lower=1> M_nD;					#nº basis functions * nº dimensions (M*D)
	int indices[M_nD,D];				#indices of combinations of basis functions 
	
	int<lower=1> Nsex1;					#nº sex1 observations
	int<lower=1> sex1[Nsex1];			#sex1 observations
}

transformed data {
	matrix[Nobs+Ncens,M_nD] PHI;
	# matrix[Nx1grid,M_nD] PHI_X1;
	# matrix[Nx1grid,M_nD] PHI_X2;
	# matrix[Nx1grid,M_nD] PHI_X3;
	# matrix[Nx1grid,M_nD] PHI_X2_1;
	# matrix[Nx1grid,M_nD] PHI_X2_2;
	
	for (m in 1:M_nD){ 
	
		#2D
		# PHI[,m] = phi_2D(L, L, indices[m,1], indices[m,2], x[,1], x[,2]); 
	
		#nD
		PHI[,m] = phi_nD(L, indices[m,], X); 
		# PHI_X1[,m] = phi_nD(L, indices[m,], X1);
		# PHI_X2[,m] = phi_nD(L, indices[m,], X2);
		# PHI_X3[,m] = phi_nD(L, indices[m,], X3);
		# PHI_X2_1[,m] = phi_nD(L, indices[m,], X2_1);
		# PHI_X2_2[,m] = phi_nD(L, indices[m,], X2_2);
	}
}

parameters {
	# row_vector<lower=0>[D] rho1;  	#multiple lengthscale
	# row_vector<lower=0>[D] rho2;  	#multiple lengthscale
	real<lower=0,upper=10> rho1;  			#one lengthscale
	real<lower=0,upper=10> rho2;  			#one lengthscale
	
	vector[M_nD] beta;
	real<lower=0> sigma;
	real<lower=0> alpha[2];
	
	real c0;
}

transformed parameters{
	vector[Nobs+Ncens] f;
	vector[Nsex1] f_sex1;
		
	matrix[M_nD,2] diagSPD;
	matrix[M_nD,2] SPD_beta;

	for(m in 1:M_nD){ 
	
		#2D
		# diagSPD[m] =  sqrt(spd_2D(alpha, rho[1], rho[2], sqrt(lambda(L, indices[m,1])), sqrt(lambda(L, indices[m,2]))));
		
		#nD
		# diagSPD[m,1] =  sqrt(spd_nD(alpha[1], rho1, sqrt(lambda_nD(L, indices[m,], D)), D)); 
		# diagSPD[m,2] =  sqrt(spd_nD(alpha[2], rho2, sqrt(lambda_nD(L, indices[m,], D)), D)); 
		
		#nD with just one lengthscale
		diagSPD[m,1] =  sqrt(spd_nD_onelscale(alpha[1], rho1, sqrt(lambda_nD(L, indices[m,], D)), D));
		diagSPD[m,2] =  sqrt(spd_nD_onelscale(alpha[2], rho2, sqrt(lambda_nD(L, indices[m,], D)), D));

	}
	
	SPD_beta[,1] = diagSPD[,1] .* beta;
	SPD_beta[,2] = diagSPD[,2] .* beta;
	
	f= c0 + PHI * SPD_beta[,1];
	f_sex1= PHI[sex1,] * SPD_beta[,2];
	f[sex1]= f[sex1] + f_sex1;
}

model{
	beta ~ normal(0,1);
	rho1 ~ inv_gamma(2,4); #normal(0,5); #gamma(2,0.5); inv_gamma(1.05,3); #
	rho2 ~ inv_gamma(2,4); #normal(0,5); #gamma(2,0.5); #inv_gamma(1.05,3);
	alpha ~ normal(0,5);
	sigma ~ normal(0,5);
	c0 ~ normal(0,5);
	
	target += normal_lpdf(y[obs_train] | f[obs_train], sigma);
	target += normal_lccdf(y[cens] | f[cens], sigma); 
}


# generated quantities{
	# vector[Nx1grid] f1;
	# vector[Nx1grid] f1_sex1;
	# vector[Nx1grid] f2;
	# vector[Nx1grid] f2_sex1;
	# vector[Nx1grid] f3;
	# vector[Nx1grid] f3_sex1;
	
	# vector[Nx1grid] f2_1;
	# vector[Nx1grid] f2_1_sex1;
	# vector[Nx1grid] f2_2;
	# vector[Nx1grid] f2_2_sex1;
	
	# vector[Nobs+Ncens] lpd;
	
	# vector[Nobs+Ncens] y_predict;
	# vector[Nx1grid] y_predict1;
	# vector[Nx1grid] y_predict2;
	# vector[Nx1grid] y_predict3;
	# vector[Nx1grid] y_predict1_sex1;
	# vector[Nx1grid] y_predict2_sex1;
	# vector[Nx1grid] y_predict3_sex1;
	# vector[Nx1grid] y_predict2_1;
	# vector[Nx1grid] y_predict2_1_sex1;
	# vector[Nx1grid] y_predict2_2;
	# vector[Nx1grid] y_predict2_2_sex1;
	# vector[Ntest] lpd;
	
	# for(i in 1:Nx1grid){
		# f1[i]= c0 + PHI_X1[i,] * SPD_beta[,1];
		# f1_sex1[i]= f1[i] + PHI_X1[i,] * SPD_beta[,2];

		# f2[i]= c0 + PHI_X2[i,] * SPD_beta[,1];
		# f2_sex1[i]= f2[i] + PHI_X2[i,] * SPD_beta[,2];
		
		# f3[i]= c0 + PHI_X3[i,] * SPD_beta[,1];
		# f3_sex1[i]= f3[i] + PHI_X3[i,] * SPD_beta[,2];
		
		# f2_1[i]= c0 + PHI_X2_1[i,] * SPD_beta[,1];
		# f2_1_sex1[i]= f2_1[i] + PHI_X2_1[i,] * SPD_beta[,2];
		
		# f2_2[i]= c0 + PHI_X2_2[i,] * SPD_beta[,1];
		# f2_2_sex1[i]= f2_2[i] + PHI_X2_2[i,] * SPD_beta[,2];
	# }
	
	# for(i in 1:(Nobs+Ncens)){
		# y_predict[i] = normal_rng(f[i], 0.0001);
	# }

	# for(i in 1:Nx1grid){
		# y_predict1[i] = normal_rng(f1[i], 0.0001);
		# y_predict2[i] = normal_rng(f2[i], 0.0001);
		# y_predict3[i] = normal_rng(f3[i], 0.0001);
		# y_predict1_sex1[i] = normal_rng(f1_sex1[i], 0.0001);
		# y_predict2_sex1[i] = normal_rng(f2_sex1[i], 0.0001);
		# y_predict3_sex1[i] = normal_rng(f3_sex1[i], 0.0001);
		# y_predict2_1[i] = normal_rng(f2_1[i], 0.0001);
		# y_predict2_1_sex1[i] = normal_rng(f2_1_sex1[i], 0.0001);
		# y_predict2_2[i] = normal_rng(f2_2[i], 0.0001);
		# y_predict2_2_sex1[i] = normal_rng(f2_2_sex1[i], 0.0001);
	# }

	# for(i in 1:(Nobs+Ncens)){
		# lpd[i] = normal_lpdf(y[i] | f[i], 0.0001);
	# }
# }


