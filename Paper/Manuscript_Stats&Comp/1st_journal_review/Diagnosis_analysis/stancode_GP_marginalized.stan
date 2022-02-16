functions {

	// SE kernel
	matrix ker_SE(real[] x, real sdgp, real lscale, real noise) { 
		int n= size(x);
		matrix[n, n] cov;
		
		cov = gp_exp_quad_cov(x, sdgp, lscale);
		
		for (i in 1:n) {
			cov[i, i] = cov[i, i] + square(noise);
		}
		return cholesky_decompose(cov);
	}
	
	// Mattern kernel
	matrix ker_mattern(real[] x, real sdgp, real lscale, real noise, int v) {
		int n= size(x);	
		matrix[n, n] cov;
				
		// Mattern 3/2
		if (v==3) {
			for (i in 1:n) {
				for (j in 1:n) {
					cov[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x[i]-x[j])))/lscale) * exp(-sqrt(3)*(fabs(x[i]-x[j]))/lscale);
				}
			}
		}
		
		// Mattern 5/2
		if (v==5) {
			for (i in 1:n) {
				for (j in 1:n) {
					cov[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x[i]-x[j])))/lscale + (5*(fabs(x[i]-x[j]))^2)/(3*lscale^2)) * exp(-sqrt(5)*(fabs(x[i]-x[j]))/lscale);
				}
			}
		}
		
		for (i in 1:n) {
			cov[i, i] = cov[i, i] + square(noise);
		}
		return cholesky_decompose(cov);
	}
	
	// Analytical GP-SE predictions
	vector gp_SE_rng(real[] x2,
					 vector y1, real[] x1,
					 real gpscale, real lscale, real noise, real delta) {
	int N1 = rows(y1);
	int N2 = size(x2);
	vector[N2] f2;
	{
	  matrix[N1, N1] K =   gp_exp_quad_cov(x1, gpscale, lscale)
						 + diag_matrix(rep_vector(square(noise), N1));
	  matrix[N1, N1] L_K = cholesky_decompose(K);

	  vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
	  vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
	  matrix[N1, N2] k_x1_x2 = gp_exp_quad_cov(x1, x2, gpscale, lscale);
	  vector[N2] f2_mu = (k_x1_x2' * K_div_y1);
	  matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
	  matrix[N2, N2] cov_f2 =   cov_exp_quad(x2, gpscale, lscale) - v_pred' * v_pred
							  + diag_matrix(rep_vector(delta, N2));
	  f2 = multi_normal_rng(f2_mu, cov_f2);
	}
	return f2;
	}

	// Analytical GP-Matern predictions
	vector gp_Matern_rng(real[] x2, vector y1, real[] x1, real gpscale, real lscale, real noise, real delta, real v) {
	int N1 = rows(y1);
	int N2 = size(x2);
	vector[N2] f2;
	matrix[N1, N1] K;
	matrix[N1, N2] k_x1_x2;
	matrix[N2, N2] cov_f2;
	matrix[N1, N1] L_K;
	vector[N1] L_K_div_y1;
	vector[N1] K_div_y1;
	vector[N2] f2_mu;
	matrix[N1, N2] v_pred;
	{
	  for (i in 1:N1) {
		for (j in i:N1) {
			if (v==3) {
			K[i,j] = gpscale^2 * (1 + (sqrt(3)*(fabs(x1[i]-x1[j])))/lscale) * exp(-sqrt(3)*(fabs(x1[i]-x1[j]))/lscale);
			}
			if (v==5) {
			K[i,j] = gpscale^2 * (1 + (sqrt(3)*(fabs(x1[i]-x1[j])))/lscale + (5*(fabs(x1[i]-x1[j]))^2)/(3*lscale^2)) * exp(-sqrt(5)*(fabs(x1[i]-x1[j]))/lscale);
			}
		}
	  }
	  for (i in 1:N1) {
	     K[i,i] = K[i, i] + square(noise);
	     for (j in (i+1):N1) {
			K[j, i] = K[i, j];
		 }
	  }
	  
	  L_K = cholesky_decompose(K);
	  L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
	  K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';

	  for (i in 1:N1) {
		for (j in 1:N2) {
			if (v==3) {
			k_x1_x2[i,j] = gpscale^2 * (1 + (sqrt(3)*(fabs(x1[i]-x2[j])))/lscale) * exp(-sqrt(3)*(fabs(x1[i]-x2[j]))/lscale);
			}
			if (v==5) {
			k_x1_x2[i,j] = gpscale^2 * (1 + (sqrt(3)*(fabs(x1[i]-x2[j])))/lscale + (5*(fabs(x1[i]-x2[j]))^2)/(3*lscale^2)) * exp(-sqrt(5)*(fabs(x1[i]-x2[j]))/lscale);
			}
		}
	  }
	  
	  f2_mu = (k_x1_x2' * K_div_y1); //'
	  v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
	  
	  for (i in 1:N2) {
		for (j in i:N2) {
			if (v==3) {
			cov_f2[i,j] = gpscale^2 * (1 + (sqrt(3)*(fabs(x2[i]-x2[j])))/lscale) * exp(-sqrt(3)*(fabs(x2[i]-x2[j]))/lscale);
			}
			if (v==5) {
			cov_f2[i,j] = gpscale^2 * (1 + (sqrt(3)*(fabs(x2[i]-x2[j])))/lscale + (5*(fabs(x2[i]-x2[j]))^2)/(3*lscale^2)) * exp(-sqrt(5)*(fabs(x2[i]-x2[j]))/lscale);
			}
		}
	  }
	  for (i in 1:N2) {
	     for (j in (i+1):N2) {
			cov_f2[j, i] = cov_f2[i, j];
		 }
	  }	  
	  
	  cov_f2 =   cov_f2 - v_pred' * v_pred + diag_matrix(rep_vector(delta, N2)); //'
	  f2 = multi_normal_rng(f2_mu, cov_f2);
	}
	return f2;
	}
	
}

data {
	int<lower=1> N;			
	real x[N];			
	vector[N] y;	
	int param_v;	
}

transformed data{
	vector[N] zeros = rep_vector(0, N);
}

parameters {
	real<lower=0> lscale;
	real<lower=0> noise;
	real<lower=0> gpscale;
}

transformed parameters{
	matrix[N,N] L_K;

	// L_K = ker_mattern(x, gpscale, lscale, noise, param_v);
	L_K = ker_SE(x, gpscale, lscale, noise);	
}

model{
	// lscale ~ normal(0, 4);
	// noise ~ normal(0, 1);
	// gpscale ~ normal(0, 10);
	
	lscale ~ gamma(1.2,0.5);
	noise ~ normal(0, 1);
	gpscale ~ normal(0, 3);
	
	y ~ multi_normal_cholesky(zeros, L_K); 
}

generated quantities{
	vector[N] f;
	f = gp_SE_rng(x, y, x, gpscale, lscale, noise, 1e-12);
}
