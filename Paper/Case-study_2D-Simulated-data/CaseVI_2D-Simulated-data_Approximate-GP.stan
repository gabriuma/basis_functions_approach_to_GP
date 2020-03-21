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
	int<lower=1> D;
	real L[D];
	int<lower=1> M;
	int<lower=1> M_nD;
	int<lower=1> Nsample;
	int Npred;
	matrix[Nsample,D] x;
	int vv_sample[Nsample];
	vector[Nsample] y;
	matrix[Npred,D] x_grid;
	vector[Npred] y_grid;
	int indices[M_nD,D];
}

transformed data {
	matrix[Npred,M_nD] PHI;
	
	for (m in 1:M_nD){ 
	
		#2D
		# PHI[,m] = phi_2D(L, L, indices[m,1], indices[m,2], x[,1], x[,2]); 
	
		#nD
		PHI[,m] = phi_nD(L, indices[m,], x_grid); 
	}
}

parameters {
	vector[M_nD] beta;
	row_vector<lower=0>[D] rho;
	real<lower=0> sigma;
	real<lower=0> alpha;
}

transformed parameters{
	vector[Nsample] f;
	
	vector[M_nD] diagSPD;
	vector[M_nD] SPD_beta;

	for(m in 1:M_nD){ 
	
		#2D
		# diagSPD[m] =  sqrt(spd_2D(alpha, rho[1], rho[2], sqrt(lambda(L, indices[m,1])), sqrt(lambda(L, indices[m,2]))));
		
		#nD
		diagSPD[m] =  sqrt(spd_nD(alpha, rho, sqrt(lambda_nD(L, indices[m,], D)), D)); 
	}
	
	SPD_beta = diagSPD .* beta;
	
	f= PHI[vv_sample,] * SPD_beta;
}

model{
	beta ~ normal(0,1);
	rho ~ inv_gamma(2,.5);
	sigma ~ normal(0,1);
	alpha ~ normal(0,2);
	
	y ~ normal(f, sigma); 
}

generated quantities{
	vector[Npred] f_grid;
	vector[Npred] y_grid_predict;

	f_grid= PHI * SPD_beta;

	for(i in 1:Npred){
		y_grid_predict[i] = normal_rng(f_grid[i], sigma);
	}
}


