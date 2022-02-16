functions {
	real lambda(real L, int m) {
		real lam;
		lam = ((m*pi())/(2*L))^2;
		return lam;
	}
	real spd_SE(real gpscale, real lscale, real w) {
		real S;
		S = (gpscale^2) * sqrt(2*pi()) * lscale * exp(-0.5*(lscale^2)*(w^2));
		return S;
	}
	real spd_Matern(real gpscale, real lscale, real w, int v) {
		real S;
		//Mattern 3/2
		if (v==3)
			S = 4*gpscale^2 * (sqrt(3)/lscale)^3 * 1/((sqrt(3)/lscale)^2 + w^2)^2;
		//Mattern 5/2
		if (v==5)
			S = 16.0/3*gpscale^2 * (sqrt(5)/lscale)^5 * 1/((sqrt(5)/lscale)^2 + w^2)^3;
		return S;
	}
	vector phi(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
		return fi;
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
	  PHI_train[,m] = phi(L, m, x_pred[vv_train]);// - mean(phi(L, m, x_pred[vv_train]));
	  PHI_pred[,m] = phi(L, m, x_pred);// - mean(phi(L, m, x_pred));	
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
	//Spectral densities for f
	for(m in 1:M){ 
		diagSPD[m] =  sqrt(spd_Matern(gpscale, lscale, sqrt(lambda(L, m)), param_v)); 
		// diagSPD[m] =  sqrt(spd_SE(gpscale, lscale, sqrt(lambda(L, m))));
	}
	SPD_beta = diagSPD .* beta;
	f= PHI_train[,] * SPD_beta;
}
model{
	lscale ~ gamma(2.5, 10); //gamma(3.75, 25); //normal(0,1); //gamma(3, 5); //
	noise ~ gamma(1, 1); //normal(0,1); //
	gpscale ~ gamma(5, 5); //normal(0,1); //
	beta ~ normal(0, 1);
	// intercept ~ normal(0, 2);
	target += normal_lpdf(y[vv_train] | f, noise);
}
generated quantities{
	vector[N_pred] f_pred;
	vector[N_pred] lpd_f;
	vector[N_pred] lpd_y;
	f_pred= PHI_pred[,] * SPD_beta;
	for(i in 1:N_pred){
		lpd_y[i] = normal_lpdf(y[i] | f_pred[i], noise);
		lpd_f[i] = normal_lpdf(f_true[i] | f_pred[i], noise);
	}
}
