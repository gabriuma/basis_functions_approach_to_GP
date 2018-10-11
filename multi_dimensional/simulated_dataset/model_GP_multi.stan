functions {
	#GP 2D (analytical predicting)
	vector gp12_pred_rng(real[] x1_grid, real[] x2_grid,
					 vector y1, real[] x1, real[] x2,
					 real alpha, real rho1, real rho2, real sigma, real delta) {
		int N1 = rows(y1);
		int N2 = size(x1_grid);
		vector[N2] f2;
		{
		  matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho1).*cov_exp_quad(x2, 1, rho2)
							 + diag_matrix(rep_vector(square(sigma), N1));
		  matrix[N1, N1] L_K = cholesky_decompose(K);

		  vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
		  vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
		  matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x1_grid, alpha, rho1).*cov_exp_quad(x2, x2_grid, 1, rho2);
		  vector[N2] f2_mu = (k_x1_x2' * K_div_y1); //'
		  matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
		  matrix[N2, N2] cov_f2 =   cov_exp_quad(x1_grid, alpha, rho1).*cov_exp_quad(x2_grid, 1, rho2) - v_pred' * v_pred
								  + diag_matrix(rep_vector(delta, N2)); //'
		  f2 = multi_normal_rng(f2_mu, cov_f2);
		}
		return f2;
	}
	
	#Additive-2D GPs (analytical predicting)
	vector gp1gp2_pred_rng(real[] x1_grid, real[] x2_grid,
					 vector y1, real[] x1, real[] x2,
					 real alpha1, real alpha2, real rho1, real rho2, real sigma, real delta) {
		int N1 = rows(y1);
		int N2 = size(x1_grid);
		vector[N2] f2;
		{
		  matrix[N1, N1] K = cov_exp_quad(x1, alpha1, rho1) + cov_exp_quad(x2, alpha2, rho2) + diag_matrix(rep_vector(square(sigma), N1));
		  matrix[N1, N1] L_K = cholesky_decompose(K);

		  vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
		  vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
		  matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x1_grid, alpha1, rho1) + cov_exp_quad(x2, x2_grid, alpha2, rho2);
		  vector[N2] f2_mu = (k_x1_x2' * K_div_y1); //'
		  matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
		  matrix[N2, N2] cov_f2 = cov_exp_quad(x1_grid, alpha1, rho1) + cov_exp_quad(x2_grid, alpha2, rho2) - v_pred' * v_pred
								  + diag_matrix(rep_vector(delta, N2)); //'
		  f2 = multi_normal_rng(f2_mu, cov_f2);
		}
		return f2;
	}
	
	#GP 1D
	vector gp(real[] x, real sdgp, real lscale, vector zgp) { 
		matrix[size(x), size(x)] cov;
		cov = cov_exp_quad(x, sdgp, lscale);
		for (n in 1:size(x)) {
			cov[n, n] = cov[n, n] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}

	#Additive-2D GPs
	vector gp1gp2(real[] x1, real[] x2, real sdgp1, real sdgp2, real lscale1, real lscale2, vector zgp) { 
		matrix[size(x1), size(x1)] cov;
		cov = cov_exp_quad(x1, sdgp1, lscale1) + cov_exp_quad(x2, sdgp2, lscale2);
		for (n in 1:size(x1)) {
			cov[n, n] = cov[n, n] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
	
	#GP 2D
	vector gp12(real[] x1, real[] x2, real sdgp, real lscale1, real lscale2, vector zgp) { 
		matrix[size(x1), size(x1)] cov;
		cov = cov_exp_quad(x1, sdgp, lscale1).*cov_exp_quad(x2, 1, lscale2);
		for (n in 1:size(x1)) {
			cov[n, n] = cov[n, n] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
}

data {
	int<lower=1> N1;			#nº training observations 
	int Npred;					#nº predicting grid observations
	int<lower=1> D;				#nº dimensions
	vector[D] x[N1];			#matrix of training inputs 
	vector[N1] y;				#vector of observations
	vector[D] x_grid[Npred];	#matrix of predicting grid inputs
	vector[Npred] y_grid;		#vector of predicting grid observations
}

transformed data{
	vector[N1] x_1; 			#for product in linear component
	vector[N1] x_2;

	for(i in 1:N1){
		x_1[i] = x[i][1]; 
		x_2[i] = x[i][2]; }
}

parameters {
	real<lower=0> rho[D];
	real<lower=0> sigma;
	real<lower=0> alpha[D];
	matrix[N1,D] eta;
	# real c0;					#coefficients in linear components
	# real c1;
	# real c2;
}

transformed parameters{
	vector[N1] f;
	
	#GP 2D
	# f= gp12(x[,1], x[,2], alpha[1], rho[1], rho[2], eta[,1]);
	
	#Additive-2D GPs
	f= gp1gp2(x[,1], x[,2], alpha[1], alpha[2], rho[1], rho[2], eta[,1]);
}

model{
	eta[,1] ~ normal(0,1);
	eta[,2] ~ normal(0,1);
	rho ~ normal(0,2);
	sigma ~ normal(0,1);
	alpha ~ normal(0,1);
	# c0 ~ normal(0,1);			#coefficients in linear components
	# c1 ~ normal(0,1);
	# c2 ~ normal(0,1);
	
	y[1:N1] ~ normal(f[1:N1], sigma); 
}

generated quantities{
	vector[Npred] f_grid;
	vector[Npred] y_grid_predict;
	vector[Npred] log_y_grid_predict;
	vector[N1] y_predict;
	vector[N1] log_y_predict;	

	#GP 2D (Analytical prediction)
	# f_grid = gp12_pred_rng(x_grid[,1], x_grid[,2], y, x[,1], x[,2], alpha[1], rho[1], rho[2], sigma, 1e-10);
	
	#Additive-2D GPs (Analytical prediction)
	f_grid = gp1gp2_pred_rng(x_grid[,1], x_grid[,2], y, x[,1], x[,2], alpha[1], alpha[2], rho[1], rho[2], sigma, 1e-10);
	
	
	for (i in 1:Npred){
		y_grid_predict[i] = normal_rng(f_grid[i], sigma); 
		log_y_grid_predict[i] = normal_lpdf(y_grid[i] | f_grid[i], sigma);
	}

	for(i in 1:N1){
		y_predict[i] = normal_rng(f[i], sigma);
		log_y_predict[i] = normal_lpdf(y[i] | f[i], sigma);
	}

}


