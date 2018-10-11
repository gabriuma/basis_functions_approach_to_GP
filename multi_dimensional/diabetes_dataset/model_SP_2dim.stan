# generated with brms 1.7.0
functions { 
} 
data { 
	int<lower=1> N1; 						# n? training observations
	int<lower=1> N2;						# n? test observations
	int ind1[N1];							# indices for training observations
	int ind2[N2];							# indices for test observations
	int<lower=1> Npred;  					# n? grid-predicting observations 
	
	int Y[N1+N2+Npred];  					# response variable 
  
	int<lower=1> K;  						# number of population-level effects 
	matrix[N1+N2+Npred, K] X;  				# population-level design matrix 
  
	# data of smooth t2(x1,x2,k=kn[j])
	int nb_1;  								# number of bases 
	int knots_1[nb_1]; 
	matrix[N1+N2+Npred, knots_1[1]] Zs_1_1; 
	matrix[N1+N2+Npred, knots_1[2]] Zs_1_2; 
	matrix[N1+N2+Npred, knots_1[3]] Zs_1_3; 
	
	int trials[N1+N2+Npred];  				# number of trials 
	int prior_only;  						# should the likelihood be ignored? 
} 
transformed data { 
	int Kc; 
	matrix[N1+N2+Npred, K - 1] Xc;  		# centered version of X 
	vector[K - 1] means_X;  				# column means of X before centering 
	Kc = K - 1;  							# the intercept is removed from the design matrix 
	for (i in 2:K) { 
		means_X[i - 1] = mean(X[, i]); 
		Xc[, i - 1] = X[, i] - means_X[i - 1]; 
	} 
} 
parameters { 
	vector[Kc] b;  						# population-level effects 
	real temp_Intercept;  				# temporary intercept 
	
	# parameters of smooth t2(x1,x2,k=kn[j])
	vector[knots_1[1]] zs_1_1; 
	real<lower=0> sds_1_1; 
	vector[knots_1[2]] zs_1_2; 
	real<lower=0> sds_1_2; 
	vector[knots_1[3]] zs_1_3; 
	real<lower=0> sds_1_3; 
} 
transformed parameters { 
	vector[N1+N2+Npred] mu; 
	vector[knots_1[1]] s_1_1; 
	vector[knots_1[2]] s_1_2; 
	vector[knots_1[3]] s_1_3; 
	s_1_1 = sds_1_1 * zs_1_1; 
	s_1_2 = sds_1_2 * zs_1_2; 
	s_1_3 = sds_1_3 * zs_1_3; 
		
	mu = Xc * b + Zs_1_1 * s_1_1 + Zs_1_2 * s_1_2 + Zs_1_3 * s_1_3 + temp_Intercept; 
} 
model { 
	# prior specifications 
	zs_1_1 ~ normal(0, 1); 
	zs_1_2 ~ normal(0, 1); 
	zs_1_3 ~ normal(0, 1); 
	sds_1_1 ~ student_t(3, 0, 10); 
	sds_1_2 ~ student_t(3, 0, 10); 
	sds_1_3 ~ student_t(3, 0, 10); 
	
	# likelihood contribution 
	if (!prior_only) { 
		Y[ind1] ~ binomial_logit(trials[ind1], mu[ind1]); 
	} 
} 
generated quantities { 
	real b_Intercept;  					# population-level intercept 
	vector[N1+N2+Npred] y_predict;
	vector[N1+N2+Npred] f_invlogit;
	vector[N2] log_y_predict;	

	b_Intercept = temp_Intercept - dot_product(means_X, b); 

	for(i in 1:(N1+N2+Npred)){
		y_predict[i] = binomial_rng(1, inv_logit(mu[i]));		
		f_invlogit[i] = inv_logit(mu[i]);
	}
	
	for(i in 1:N2){
		log_y_predict[i] = binomial_logit_lpmf(Y[ind2[i]] | 1, mu[ind2[i]]);
	}
} 
