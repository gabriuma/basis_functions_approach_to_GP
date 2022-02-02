functions {

	real lambda(real L, int m) {
		real lam;
		lam = ((m*pi())/(2*L))^2;
				
		return lam;
	}
	
	real spd_SE(real gpscale, real lscale, real w) {
		real S;
		S = (gpscale^2) * sqrt(2*pi()) * lscale * exp(-0.5*(lscale^2)*(w^2));
		//if(S <= 0.001) S = 0.001;

		return S;
	}
	
	real spd_Matern(real gpscale, real lscale, real w, int v) {
		real S;
		
		// Mattern 3/2
		if (v==3) {
			S = 4*gpscale^2 * (sqrt(3)/lscale)^3 * 1/((sqrt(3)/lscale)^2 + w^2)^2;
		}
		// Mattern 5/2
		if (v==5) {	
			S = 16.0/3*gpscale^2 * (sqrt(5)/lscale)^5 * 1/((sqrt(5)/lscale)^2 + w^2)^3;
		}
		return S;
	}
	
	vector phi(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
				
		return fi;
	}

}

data {
	int<lower=1> N;
	vector[N] x;			
	vector[N] y;	
	// int param_v;	

	real L;	
	int<lower=1> M;	
}

transformed data{
	//Basis functions for f
	matrix[N,M] PHI_sample;

	for (m in 1:M){ PHI_sample[,m] = phi(L, m, x); }
}

parameters {
	real<lower=0> lscale;
	real<lower=0> noise;
	real<lower=0> gpscale;
	
	vector[M] beta;
}

transformed parameters{
	vector[N] f;
	vector[M] diagSPD;
	vector[M] SPD_beta;
	
	//Spectral densities for f
	for(m in 1:M){ 
		// diagSPD[m] =  sqrt(spd_Matern(gpscale, lscale, sqrt(lambda(L, m)), param_v)); 
		diagSPD[m] =  sqrt(spd_SE(gpscale, lscale, sqrt(lambda(L, m))));
	}
	
	SPD_beta = diagSPD .* beta;

	f= PHI_sample[,] * SPD_beta;
	
}

model{
	lscale ~ gamma(1.2,0.5);
	noise ~ normal(0, 1);
	gpscale ~ normal(0, 3);
	
	beta ~ normal(0, 1);
	
	y ~ normal(f, noise); 
}

generated quantities{
	vector[N] lpd;
	for(i in 1:N){
		lpd[i] = normal_lpdf(y[i] | f[i], noise);
	}
}
