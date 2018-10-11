# generated with brms 1.7.0
functions { 
} 
data { 
	int<lower=1> N1; 					# number of training observations
	int<lower=1> Npred;					# number of predicting/test observations
	
	vector[N1+Npred] Y;  				# response variable 
	
	int<lower=1> K;  					# number of population-level effects 
	matrix[N1+Npred, K] X;  					# population-level design matrix 

	# data of smooth t2(x1,x2)
	int nb_1;  							# number of bases 
	int knots_1[nb_1]; 
	matrix[N1+Npred, knots_1[1]] Zs_1_1; 
	matrix[N1+Npred, knots_1[2]] Zs_1_2; 
	matrix[N1+Npred, knots_1[3]] Zs_1_3; 
	
	int prior_only;  					# should the likelihood be ignored? 
} 

transformed data { 
	int Kc; 
	matrix[N1+Npred, K - 1] Xc;  				# centered version of X 
	vector[K - 1] means_X; 				# column means of X before centering 
	Kc = K - 1;  						# the intercept is removed from the design matrix 
	for (i in 2:K) { 
	means_X[i - 1] = mean(X[, i]); 
	Xc[, i - 1] = X[, i] - means_X[i - 1]; 
	} 
} 
parameters { 
	vector[Kc] b;  						# population-level effects 
	real temp_Intercept;  				# temporary intercept 
	
	# parameters of smooth t2(x1,x2)
	vector[knots_1[1]] zs_1_1; 
	real<lower=0> sds_1_1; 
	vector[knots_1[2]] zs_1_2; 
	real<lower=0> sds_1_2; 
	vector[knots_1[3]] zs_1_3; 
	real<lower=0> sds_1_3; 
	
	real<lower=0> sigma;  				# residual SD 
} 
transformed parameters { 
	vector[N1+Npred] mu; 
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
	sigma ~ student_t(3, 0, 10); 
	
	# likelihood contribution 
	if (!prior_only) { 
		Y[1:N1] ~ normal(mu[1:N1], sigma); 
	} 
} 
generated quantities { 
	real b_Intercept;  					# population-level intercept 
	vector[N1+Npred] y_predict;
	vector[N1+Npred] log_y_predict;	

	b_Intercept = temp_Intercept - dot_product(means_X, b); 

	for(i in 1:(N1+Npred)){
		y_predict[i] = normal_rng(mu[i], sigma);

		log_y_predict[i] = normal_lpdf(Y[i] | mu[i], sigma);
	}
}

 