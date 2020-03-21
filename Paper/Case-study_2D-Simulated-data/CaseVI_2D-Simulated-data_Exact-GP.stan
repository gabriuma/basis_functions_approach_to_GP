functions {
	#GP 2D (analytical predicting)
	vector gp12_pred_rng(real[] x1_grid, real[] x2_grid,
					 vector y1, real[] x1, real[] x2,
					 real alpha, real rho1, real rho2, real sigma, real delta) {
		int Nsample = rows(y1);
		int N2 = size(x1_grid);
		vector[N2] f2;
		{
		  matrix[Nsample, Nsample] K =   cov_exp_quad(x1, alpha, rho1).*cov_exp_quad(x2, 1, rho2)
							 + diag_matrix(rep_vector(square(sigma), Nsample));
		  matrix[Nsample, Nsample] L_K = cholesky_decompose(K);

		  vector[Nsample] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
		  vector[Nsample] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
		  matrix[Nsample, N2] k_x1_x2 = cov_exp_quad(x1, x1_grid, alpha, rho1).*cov_exp_quad(x2, x2_grid, 1, rho2);
		  vector[N2] f2_mu = (k_x1_x2' * K_div_y1); //'
		  matrix[Nsample, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
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
		int Nsample = rows(y1);
		int N2 = size(x1_grid);
		vector[N2] f2;
		{
		  matrix[Nsample, Nsample] K = cov_exp_quad(x1, alpha1, rho1) + cov_exp_quad(x2, alpha2, rho2) + diag_matrix(rep_vector(square(sigma), Nsample));
		  matrix[Nsample, Nsample] L_K = cholesky_decompose(K);

		  vector[Nsample] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
		  vector[Nsample] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
		  matrix[Nsample, N2] k_x1_x2 = cov_exp_quad(x1, x1_grid, alpha1, rho1) + cov_exp_quad(x2, x2_grid, alpha2, rho2);
		  vector[N2] f2_mu = (k_x1_x2' * K_div_y1); //'
		  matrix[Nsample, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
		  matrix[N2, N2] cov_f2 = cov_exp_quad(x1_grid, alpha1, rho1) + cov_exp_quad(x2_grid, alpha2, rho2) - v_pred' * v_pred
								  + diag_matrix(rep_vector(delta, N2)); //'
		  f2 = multi_normal_rng(f2_mu, cov_f2);
		}
		return f2;
	}
	
	# kernel 1D
	matrix ker_SE(real[] x, real sdgp, real lscale, real sigma) { 
		int n= size(x);
		matrix[n, n] cov;
		
		cov = cov_exp_quad(x, sdgp, lscale);
		
		for (i in 1:n) {
			cov[i, i] = cov[i, i] + square(sigma);
		}
		return cholesky_decompose(cov);
	}

	#Additive-2D kernel
	matrix ker_gp1gp2(real[] x1, real[] x2, real sdgp1, real sdgp2, real lscale1, real lscale2, real sigma) { 
		matrix[size(x1), size(x1)] cov;
		cov = cov_exp_quad(x1, sdgp1, lscale1) + cov_exp_quad(x2, sdgp2, lscale2);
		for (n in 1:size(x1)) {
			cov[n, n] = cov[n, n] + square(sigma);
		}
		return cholesky_decompose(cov);
	}
	
	#Kernel 2D
	matrix ker_gp12(real[] x1, real[] x2, real sdgp, real lscale1, real lscale2, real sigma) { 
		matrix[size(x1), size(x1)] cov;
		cov = cov_exp_quad(x1, sdgp, lscale1).*cov_exp_quad(x2, 1, lscale2);
		for (n in 1:size(x1)) {
			cov[n, n] = cov[n, n] + square(sigma);
		}
		return cholesky_decompose(cov);
	}
}

data {
	int<lower=1> Nsample;
	int Npred;
	int<lower=1> D;
	vector[D] x[Nsample];
	vector[Nsample] y;
	vector[D] x_grid[Npred];
	vector[Npred] y_grid;
}

transformed data{
	vector[Nsample] zeros = rep_vector(0, Nsample);
}

parameters {
	real<lower=0> rho[D];
	real<lower=0> sigma;
	real<lower=0> alpha[D];
}

transformed parameters{
	matrix[Nsample,Nsample] L_K;

	#Kernel 2D
	L_K = ker_gp12(x[,1], x[,2], alpha[1], rho[1], rho[2], sigma);
	
	#Additive-2D Kernels 
	# L_K= kernel_gp1gp2(x[,1], x[,2], alpha[1], alpha[2], rho[1], rho[2], sigma);
}

model{
	rho ~ inv_gamma(2,.5);
	sigma ~ normal(0,1);
	alpha ~ normal(0,2);
	
	y ~ multi_normal_cholesky(zeros, L_K);

}

generated quantities{
	vector[Npred] f_grid;
	vector[Npred] y_grid_predict;
	vector[Npred] log_y_grid_predict;

	#GP 2D (Analytical prediction)
	f_grid = gp12_pred_rng(x_grid[,1], x_grid[,2], y, x[,1], x[,2], alpha[1], rho[1], rho[2], sigma, 1e-10);
	
	#Additive-2D GPs (Analytical prediction)
	# f_grid = gp1gp2_pred_rng(x_grid[,1], x_grid[,2], y, x[,1], x[,2], alpha[1], alpha[2], rho[1], rho[2], sigma, 1e-10);
	
	
	for (i in 1:Npred){
		y_grid_predict[i] = normal_rng(f_grid[i], sigma); 
		log_y_grid_predict[i] = normal_lpdf(y_grid[i] | f_grid[i], sigma);
	}
}


