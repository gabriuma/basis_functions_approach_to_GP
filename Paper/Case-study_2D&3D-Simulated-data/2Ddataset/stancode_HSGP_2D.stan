functions {
  //Eigenvalue function
	real lambda_1D(real L, int m) {
		return ((m*pi())/(2*L))^2;
	}
	vector lambda_nD(vector L, row_vector m, int D) {
		vector[D] lam;
		for(i in 1:D){
			lam[i] = ((m[i]*pi())/(2*L[i]))^2; }
		return lam;
	}
	
	//Square root of spectral densitiy function (for a squared exponential kernel)
	real sqrt_spd_1D(real gpscale, real lscale, real w) {
		return gpscale * sqrt(sqrt(2*pi()) * lscale) * exp(-.25*(lscale^2)*(w^2));
	}
	real sqrt_spd_2D(real gpscale, real lscale1, real lscale2, real w1, real w2) {
		return gpscale * sqrt(sqrt(2*pi())^2 * lscale1*lscale2) * exp(-.25*(lscale1^2*w1^2 + lscale2^2*w2^2));
	}
	real sqrt_spd_nD(real gpscale, vector lscale, vector w, int D) {
		return gpscale * sqrt(sqrt(2*pi())^D * prod(lscale)) * exp(-.25*((to_row_vector(lscale) .* to_row_vector(lscale)) * (w .* w)));
	}

  //Square root vector of spectral densities (for a squared exponential kernel)
  vector sqrt_diagSPD_1D(real gpscale, real lscale, real L, int M) {
    return gpscale * sqrt(sqrt(2*pi()) * lscale) * exp(-.25*(lscale*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
  }
  vector sqrt_diagSPD_nD(real gpscale, vector lscale, vector L, matrix indices, int D) {
    return gpscale *  sqrt(sqrt(2*pi())^D * prod(lscale)) * exp(-.25 * (indices^2 * (lscale*pi() ./ (2*L))^2));
  }
  
  //Eigenfunction
	vector phi_1D(real L, int m, vector x) {
		return 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
	}
	vector phi_2D(real L1, real L2, int m1, int m2, vector x1, vector x2) {
		vector[rows(x1)] fi1;
		vector[rows(x1)] fi2;
		fi1 = 1/sqrt(L1)*sin(m1*pi()*(x1+L1)/(2*L1));
		fi2 = 1/sqrt(L2)*sin(m2*pi()*(x2+L2)/(2*L2));
		return fi1 .* fi2;
	}
	vector phi_nD(vector L, row_vector m, matrix x) {
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
	
	//Matrix of eigenfunction values
  matrix PHI_1D(int N, int M, real L, vector x) {
    matrix[N,M] PHI = sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
    return PHI;
  }
  matrix PHI_2D(int N, int M1, int M2, real L1, real L2, vector x1, vector x2) {
    matrix[N,M1*M2] PHI;
    matrix[N,M1] PHI_1 = sin(diag_post_multiply(rep_matrix(pi()/(2*L1) * (x1+L1), M1), linspaced_vector(M1, 1, M1)))/sqrt(L1);
    matrix[N,M2] PHI_2 = sin(diag_post_multiply(rep_matrix(pi()/(2*L2) * (x2+L2), M2), linspaced_vector(M2, 1, M2)))/sqrt(L2);
    PHI[,1:M2] = rep_matrix(PHI_1[,1], M2) .* PHI_2;
    for(i in 2:M1)
      PHI[,1:(M2*i)] = append_col(PHI[,1:(M2*(i-1))], rep_matrix(PHI_1[,i], M2) .* PHI_2);
    return PHI;
  }
}

data {
	int<lower=1> D;
	vector[D] L;
	int M[D];
	int<lower=1> M_nD;
	int<lower=1> N_sample;
	int N_pred;
	int vv_sample[N_sample];
	matrix[N_pred,D] x_pred;
	vector[N_pred] y_pred;
	matrix[M_nD,D] indices;
}

transformed data {
	matrix[N_pred,M_nD] PHI;
	PHI = PHI_2D(N_pred, M[1], M[2], L[1], L[2], x_pred[,1], x_pred[,2]);
}

parameters {
	vector[M_nD] beta;
	vector<lower=0>[D] lscale;
	real<lower=0> noise;
	real<lower=0> gpscale;
}

transformed parameters{
	vector[N_sample] f;
	vector[M_nD] SPD_beta;
 {
	vector[M_nD] SPD;
	SPD = sqrt_diagSPD_nD(gpscale, lscale, L, indices, D);
	SPD_beta = SPD .* beta;
	f= PHI[vv_sample,] * SPD_beta;
 }
}

model{
	beta ~ normal(0,1);
	lscale ~ inv_gamma(2,.5);
	noise ~ normal(0,2);
	gpscale ~ normal(0,4);
	target += normal_lpdf(y_pred[vv_sample]  | f, noise);
}

generated quantities{
  vector[N_pred] f_pred;
  vector[N_sample] elpd;
  f_pred= PHI * SPD_beta;
	for(i in 1:N_sample)
		elpd[i] = normal_lpdf(y_pred[vv_sample][i] | f[i], noise);
}


