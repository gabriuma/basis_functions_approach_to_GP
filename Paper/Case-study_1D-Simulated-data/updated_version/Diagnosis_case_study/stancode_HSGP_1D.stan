functions {
  //Eigenvalue function
	real lambda(real L, int m) {
		return ((m*pi())/(2*L))^2;
	}
	//Spectral densitiy function (squared exponential (QE) kernel)
	real spd_1D_QE(real gpscale, real lscale, real w) {
		return (gpscale^2) * sqrt(2*pi()) * lscale * exp(-0.5*(lscale^2)*(w^2));
	}
	//Square root of spectral densitiy function (squared exponential (QE) kernel)
	real sqrt_spd_1D_QE(real gpscale, real lscale, real w) {
		return gpscale * sqrt(sqrt(2*pi()) * lscale) * exp(-.25*(lscale^2)*(w^2));
	}
	//Spectral densitiy function (Mattern kernel)
	real spd_1D_Mattern(real gpscale, real lscale, real w, int v) {
	  real S;
		if (v==3) // Mattern 3/2
			S= 4*gpscale^2 * (sqrt(3)/lscale)^3 * 1/((sqrt(3)/lscale)^2 + w^2)^2;
		if (v==5) // Mattern 5/2
			S= 16.0/3*gpscale^2 * (sqrt(5)/lscale)^5 * 1/((sqrt(5)/lscale)^2 + w^2)^3;
		return S;
	}
	// Square root of spectral densitiy function (Mattern kernel)
	real sqrt_spd_1D_Mattern(real gpscale, real lscale, real w, int v) {
	  real S;
		if (v==3){ //Mattern 3/2
		  S= 2*gpscale * sqrt((sqrt(3)/lscale)^3) * 1/((sqrt(3)/lscale)^2 + w^2);
		}
		if (v==5){ //Mattern 5/2
			S= 4/sqrt(3)*gpscale * sqrt((sqrt(5)/lscale)^5) * 1/((sqrt(5)/lscale)^2 + w^2)^(3.0/2);
	  }
		return S;
	}
	//Square root vector of spectral densities (squared exponential (QE) kernel)
  vector sqrt_diagSPD_1D_QE(real gpscale, real lscale, real L, int M) {
    return gpscale * sqrt(sqrt(2*pi()) * lscale) * exp(-.25*(lscale*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
  }
	//Square root vector of spectral densities (Matern kernel)
	vector sqrt_diagSPD_1D_Mattern(real gpscale, real lscale, real L, int M, int v) {
	  vector[M] S;
	  if (v==3) //Mattern 3/2
      S= 2*gpscale * sqrt((sqrt(3)/lscale)^3) * 1 ./ ((sqrt(3)/lscale)^2 + (pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
	  if (v==5) //Mattern 5/2
      S= 4/sqrt(3)*gpscale * sqrt((sqrt(5)/lscale)^5) * 1 ./ ((sqrt(5)/lscale)^2 + (pi()/2/L)^2 * linspaced_vector(M, 1, M)^2)^(3.0/2);
    return S;
  }
  //Eigenfunction
	vector phi_1D(real L, int m, vector x) {
		return 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
	}
}

data {
	int<lower=1> N_pred;
	int<lower=1> N_train;
	vector[N_pred] x_pred;			
	vector[N_pred] y;
	vector[N_pred] f_true;	
	int param_v;	
	int vv_train[N_train];
	real L;	
	int<lower=1> M;	
}
transformed data{
	//Basis functions for f
	matrix[N_train,M] PHI_train;
	matrix[N_pred,M] PHI_pred;
	for (m in 1:M){
	  PHI_train[,m] = phi_1D(L, m, x_pred[vv_train]);
	  PHI_pred[,m] = phi_1D(L, m, x_pred);	
	}
}
parameters {
  // real intercept;
	real<lower=0> lscale;
	real<lower=0> noise;
	real<lower=0> gpscale;
	vector[M] beta;
}
transformed parameters{
	vector[N_train] f;
	vector[M] diagSPD;
	vector[M] SPD_beta;
	//Vector of spectral densities
	for(m in 1:M){ 
		diagSPD[m] =  sqrt_spd_1D_Mattern(gpscale, lscale, sqrt(lambda(L, m)), param_v); 
	}
	SPD_beta = diagSPD .* beta;
	f= PHI_train[,] * SPD_beta;
}
model{
	lscale ~ gamma(1.2,0.2);
	noise ~ normal(0,1);
	gpscale ~ normal(0,3);
	beta ~ normal(0, 1);
	// intercept ~ normal(0, 2);
	target += normal_lpdf(y[vv_train] | f, noise);
}
generated quantities{
	vector[N_pred] f_pred;
	vector[N_pred] elpd;
	f_pred= PHI_pred[,] * SPD_beta;
	for(i in 1:N_pred)
		elpd[i] = normal_lpdf(y[i] | f_pred[i], noise);
}
