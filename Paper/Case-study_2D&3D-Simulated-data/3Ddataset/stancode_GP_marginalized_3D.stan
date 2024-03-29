functions {
	//GP 2D (analytical predicting)
	vector gp12_pred_rng(real[] x1_grid, real[] x2_grid,
					 vector y1, real[] x1, real[] x2,
					 real gpscale, real rho1, real rho2, real sigma, real delta) {
		int N_sample = rows(y1);
		int N2 = size(x1_grid);
		vector[N2] f2;
		{
		  matrix[N_sample, N_sample] K =   gp_exp_quad_cov(x1, gpscale, rho1).*gp_exp_quad_cov(x2, 1, rho2)
							 + diag_matrix(rep_vector(square(sigma), N_sample));
		  matrix[N_sample, N_sample] L_K = cholesky_decompose(K);
		  vector[N_sample] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
		  vector[N_sample] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
		  matrix[N_sample, N2] k_x1_x2 = gp_exp_quad_cov(x1, x1_grid, gpscale, rho1).*gp_exp_quad_cov(x2, x2_grid, 1, rho2);
		  vector[N2] f2_mu = (k_x1_x2' * K_div_y1); //'
		  matrix[N_sample, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
		  matrix[N2, N2] cov_f2 =   gp_exp_quad_cov(x1_grid, gpscale, rho1).*gp_exp_quad_cov(x2_grid, 1, rho2) - v_pred' * v_pred
								  + diag_matrix(rep_vector(delta, N2)); //'
		  f2 = multi_normal_rng(f2_mu, cov_f2);
		}
		return f2;
	}
	
	//GP 3D (analytical predicting)
	vector gp123_pred_rng(real[] x1_grid, real[] x2_grid, real[] x3_grid,
					 vector y1, real[] x1, real[] x2, real[] x3,
					 real gpscale, real rho1, real rho2, real rho3, real sigma, real delta) {
		int N_sample = rows(y1);
		int N2 = size(x1_grid);
		vector[N2] f2;
		{
		  matrix[N_sample, N_sample] K =   gp_exp_quad_cov(x1, gpscale, rho1).*gp_exp_quad_cov(x2, 1, rho2).*gp_exp_quad_cov(x3, 1, rho3)
							 + diag_matrix(rep_vector(square(sigma), N_sample));
		  matrix[N_sample, N_sample] L_K = cholesky_decompose(K);
		  vector[N_sample] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
		  vector[N_sample] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
		  matrix[N_sample, N2] k_x1_x2 = gp_exp_quad_cov(x1, x1_grid, gpscale, rho1).*gp_exp_quad_cov(x2, x2_grid, 1, rho2).*gp_exp_quad_cov(x3, x3_grid, 1, rho3);
		  vector[N2] f2_mu = (k_x1_x2' * K_div_y1); //'
		  matrix[N_sample, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
		  matrix[N2, N2] cov_f2 =   gp_exp_quad_cov(x1_grid, gpscale, rho1).*gp_exp_quad_cov(x2_grid, 1, rho2).*gp_exp_quad_cov(x3_grid, 1, rho3) - v_pred' * v_pred
								  + diag_matrix(rep_vector(delta, N2)); //'
		  f2 = multi_normal_rng(f2_mu, cov_f2);
		}
		return f2;
	}
	
	//Additive GP 2D (analytical predicting)
	vector gp1gp2_pred_rng(real[] x1_grid, real[] x2_grid,
					 vector y1, real[] x1, real[] x2,
					 real alpha1, real alpha2, real rho1, real rho2, real sigma, real delta) {
		int N_sample = rows(y1);
		int N2 = size(x1_grid);
		vector[N2] f2;
		{
		  matrix[N_sample, N_sample] K = gp_exp_quad_cov(x1, alpha1, rho1) + gp_exp_quad_cov(x2, alpha2, rho2) + diag_matrix(rep_vector(square(sigma), N_sample));
		  matrix[N_sample, N_sample] L_K = cholesky_decompose(K);
		  vector[N_sample] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
		  vector[N_sample] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
		  matrix[N_sample, N2] k_x1_x2 = gp_exp_quad_cov(x1, x1_grid, alpha1, rho1) + gp_exp_quad_cov(x2, x2_grid, alpha2, rho2);
		  vector[N2] f2_mu = (k_x1_x2' * K_div_y1); //'
		  matrix[N_sample, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
		  matrix[N2, N2] cov_f2 = gp_exp_quad_cov(x1_grid, alpha1, rho1) + gp_exp_quad_cov(x2_grid, alpha2, rho2) - v_pred' * v_pred
								  + diag_matrix(rep_vector(delta, N2)); //'
		  f2 = multi_normal_rng(f2_mu, cov_f2);
		}
		return f2;
	}
	
	//Kernel 1D
	matrix ker_SE(real[] x, real sdgp, real lscale, real sigma) { 
		int n= size(x);
		matrix[n, n] cov;
		cov = gp_exp_quad_cov(x, sdgp, lscale);
		for (i in 1:n)
			cov[i, i] = cov[i, i] + square(sigma);
		return cholesky_decompose(cov);
	}

	//Additive kernel 2D
	matrix ker_gp1gp2(real[] x1, real[] x2, real sdgp1, real sdgp2, real lscale1, real lscale2, real sigma) { 
		matrix[size(x1), size(x1)] cov;
		cov = gp_exp_quad_cov(x1, sdgp1, lscale1) + gp_exp_quad_cov(x2, sdgp2, lscale2);
		for (n in 1:size(x1))
			cov[n, n] = cov[n, n] + square(sigma);
		return cholesky_decompose(cov);
	}
	
	//Kernel 2D
	matrix ker_gp12(real[] x1, real[] x2, real sdgp, real lscale1, real lscale2, real sigma) { 
		matrix[size(x1), size(x1)] cov;
		cov = gp_exp_quad_cov(x1, sdgp, lscale1).*gp_exp_quad_cov(x2, 1, lscale2);
		for (n in 1:size(x1))
			cov[n, n] = cov[n, n] + square(sigma);
		return cholesky_decompose(cov);
	}	
	
	//Kernel 3D
	matrix ker_gp123(real[] x1, real[] x2, real[] x3, real sdgp, real lscale1, real lscale2, real lscale3, real sigma) { 
		matrix[size(x1), size(x1)] cov;
		cov = gp_exp_quad_cov(x1, sdgp, lscale1).*gp_exp_quad_cov(x2, 1, lscale2).*gp_exp_quad_cov(x3, 1, lscale3);
		for (n in 1:size(x1))
			cov[n, n] = cov[n, n] + square(sigma);
		return cholesky_decompose(cov);
	}
}

data {
	int<lower=1> N_sample;
	int<lower=1> vv_sample[N_sample];
	int N_pred;
	int<lower=1> D;
	vector[D] x_pred[N_pred];
	vector[N_pred] y_pred;
}

transformed data{
	vector[N_sample] zeros = rep_vector(0, N_sample);
}

parameters {
	real<lower=0> lscale[D];
	real<lower=0> sigma;
	real<lower=0> gpscale;
}

transformed parameters{
}

model{
  matrix[N_sample,N_sample] L_K;
	lscale ~ inv_gamma(2,.5); //gamma(2,2); //normal(0,2); //
	sigma ~ normal(0,2);
	gpscale ~ normal(0,4);
	L_K = ker_gp123(x_pred[vv_sample,1], x_pred[vv_sample,2], x_pred[vv_sample,3], gpscale, lscale[1], lscale[2], lscale[3], sigma);
	y_pred[vv_sample] ~ multi_normal_cholesky(zeros, L_K);
}

generated quantities{
	vector[N_pred] f_pred;
	vector[N_sample] elpd;

	//GP 3D (Analytical prediction)
	f_pred = gp123_pred_rng(x_pred[,1], x_pred[,2], x_pred[,3], y_pred[vv_sample], x_pred[vv_sample,1], x_pred[vv_sample,2], x_pred[vv_sample,3], gpscale, lscale[1], lscale[2], lscale[3], sigma, 1e-10);
	
	for (i in 1:N_sample){
		elpd[i] = normal_lpdf(y_pred[vv_sample][i] | f_pred[vv_sample][i], sigma);
	}
}


