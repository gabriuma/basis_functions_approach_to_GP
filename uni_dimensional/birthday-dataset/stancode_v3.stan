functions {

	real lambda(real L, int m) {
		real lam;
		lam = ((m*pi())/(2*L))^2;
				
		return lam;
	}
	
	real spd_SE(real alpha, real rho, real w) {
		real S;
		S = (alpha^2) * sqrt(2*pi()) * rho * exp(-0.5*(rho^2)*(w^2));
				
		return S;
	}
	
	real spd_Matt(real alpha, real rho, real w) {
		real S;
		S = 4*alpha^2 * (sqrt(3)/rho)^3 * 1/((sqrt(3)/rho)^2 + w^2)^2;
				
		return S;
	}
	
	real q_periodic(real alpha, real rho, int v) {
		real q;
		real I;
		
		# Periodic
		if(v==0){
			I =  modified_bessel_first_kind(v, 1/rho^2); 
			q = (alpha^2) * I/exp(1/rho^2);
			return q;
		} else{
			I =  modified_bessel_first_kind(v, 1/rho^2); 
			q = (alpha^2) * 2*I/exp(1/rho^2);
			return q;
		}
		
	}
	
	vector phi_SE(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
				
		return fi;
	}
	
	vector phi_cosine_periodic(real w0, int m, vector x) {
		vector[rows(x)] fi;
		fi = cos(m*w0*x);
				
		return fi;
	}
	
	vector phi_sin_periodic(real w0, int m, vector x) {
		vector[rows(x)] fi;
		fi = sin(m*w0*x); 
		
		return fi;
	}
}

data {
	real L_f1;					#boundary value for function 1
	int<lower=1> M_f1;			#num basis functions for function 1
	int<lower=-1> J_f3;			#num cosine and sinu functions for function 3
	int<lower=-1> J_f4;			#num cosine and sinu functions for function 4
	
	int<lower=1> N;				#num observations
	vector[N] x;				#input variable
	vector[N] y;				#respose variable
	
	real period_year;			#period of the year
	real period_week;			#period of the week
}

transformed data {
	real<lower=0>  scale_global=0.04;	#scale for the half-t prior for tau
	real<lower=1>  nu_global=1;			#df for the half-t priors for tau
	real<lower=1>  nu_local=1;			#df for the half-t priors for lambdas. (nu_local= 1  corresponds  to  the  horseshoe)
	
	#Regularization horseshoe prior
	# real<lower=0>  slab_scale= 3;
	# real <lower=0> slab_df=2; 

	#Basis functions for f1, f3 and f4
	matrix[N,M_f1] PHI_f1;
	matrix[N,2*J_f3+1] PHI_f3;
	matrix[N,2*J_f4+1] PHI_f4;
	
	for (m in 1:M_f1){ PHI_f1[,m] = phi_SE(L_f1, m, x); }
	
	for (m in 0:J_f3){ PHI_f3[,m+1] = phi_cosine_periodic(2*pi()/period_year, m, x); }
	for (m in 1:J_f3){ PHI_f3[,J_f3+1+m] = phi_sin_periodic(2*pi()/period_year, m, x); }
	
	for (m in 0:J_f4){ PHI_f4[,m+1] = phi_cosine_periodic(2*pi()/period_week, m, x); }
	for (m in 1:J_f4){ PHI_f4[,J_f4+1+m] = phi_sin_periodic(2*pi()/period_week, m, x); }

}

parameters {
	#variables for the basis function models
	vector[M_f1] beta_f1;
	vector[2*J_f3+1] beta_f3;
	vector[2*J_f4+1] beta_f4;
	
	#hyperparameters
	vector<lower=0>[4] rho;
	vector<lower=0>[4] alpha;
	real<lower=0> sigma;
	
	#Horseshoe prior
	vector[366] z;
	real<lower=0>  r1_global;
	real<lower=0>  r2_global;
	vector<lower=0>[366]  r1_local;
	vector<lower=0>[366]  r2_local;
	
	#Regularization horseshoe prior
	# real<lower=0> caux;
}

transformed parameters{
	vector[N] f;
	vector[N] f1;
	vector[N] f3;
	vector[N] f4;
	vector[M_f1] diagSPD_f1;
	vector[M_f1] SPD_beta_f1;
	vector[2*J_f3+1] diagSPD_f3;
	vector[2*J_f3+1] SPD_beta_f3;
	vector[2*J_f4+1] diagSPD_f4;
	vector[2*J_f4+1] SPD_beta_f4;
	
	#Horseshoe prior effects
	real<lower=0> tau;						#global shrinkage parameter
	vector<lower=0>[366] lambda_h;			#local shrinkage parameter
	
	vector[366] f5_366;					# function f5 (horseshoe effects)
	vector[365] f5_365;					# copy of f5 removing leap day
	vector[365*3+366] f5_4years;		# f5_365 + f5_365 + f5_365 + f5_366
	vector[N] f5;						# 5 * f5_4years 
	
	#Regularization horseshoe
	# vector<lower=0>[N1] lambda_tilde;
	# real<lower=0> c;
	
	#Horseshoe
	tau = r1_global * sqrt(r2_global );
	lambda_h = r1_local .* sqrt(r2_local );
	
	f5_366 = z .* lambda_h*tau;
	
	f5_365 = append_row(f5_366[1:59],f5_366[61:366]);
	
	f5_4years = append_row(append_row(f5_365,f5_365),append_row(f5_365,f5_366));
	
	f5= append_row(append_row(
			append_row(f5_4years,f5_4years),
				append_row(f5_4years,f5_4years)),
					f5_4years);
	
	#Regularization horseshoe
	# c = slab_scale * sqrt(caux);
	# lambda_tilde = sqrt( c^2 * square(lambda_h) ./ (c^2 + tau^2* square(lambda_h )) );
	# f5 = z .*  lambda_tilde*tau;

	#Spectral densities for f1, f3 and f4
	for(m in 1:M_f1){ 
		diagSPD_f1[m] =  sqrt(spd_SE(alpha[1], rho[1], sqrt(lambda(L_f1, m)))); 
	}
	for(m in 0:J_f3){
		diagSPD_f3[m+1] =  sqrt(q_periodic(alpha[3], rho[3], m)); 
	}
	for(m in 1:J_f3){ 
		diagSPD_f3[J_f3+1+m] =  sqrt(q_periodic(alpha[3], rho[3], m)); 
	}
	for(m in 0:J_f4){
		diagSPD_f4[m+1] =  sqrt(q_periodic(alpha[4], rho[4], m)); 
	}
	for(m in 1:J_f4){ 
		diagSPD_f4[J_f4+1+m] =  sqrt(q_periodic(alpha[4], rho[4], m)); 
	}
	
	SPD_beta_f1 = diagSPD_f1 .* beta_f1;
	SPD_beta_f3 = diagSPD_f3 .* beta_f3;
	SPD_beta_f4 = diagSPD_f4 .* beta_f4;
	

	f1= PHI_f1[,] * SPD_beta_f1;
	f3= PHI_f3[,] * SPD_beta_f3;
	f4= PHI_f4[,] * SPD_beta_f4;
	
	f= f1 + f3 + f4 + f5;     
}

model{
	beta_f1 ~ normal(0,1);
	beta_f3 ~ normal(0,1);
	beta_f4 ~ normal(0,1);
	
	rho ~ normal(0,2);				#lengthscale GP
	alpha ~ normal(0,10);			#magnitud GP
	sigma ~ normal(0,1);			#noise
	
	y ~ normal(f, sigma); 
	
	#Horseshoe prior effects
	z ~ normal(0, 1);
	r1_local ~ normal (0.0,  1.0);
	r2_local ~ inv_gamma (0.5* nu_local , 0.5* nu_local );

	r1_global ~ normal (0.0,  scale_global*sigma );
	r2_global ~ inv_gamma (0.5* nu_global , 0.5* nu_global );
	
	#Regularization horseshoe prior
	# caux ~ inv_gamma(0.5* slab_df , 0.5* slab_df );	
}

generated quantities{

}
