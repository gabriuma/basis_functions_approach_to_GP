functions {
	// kernel 1D
	matrix ker_SE(real[] x, real sdgp, real lscale, real sigma) { 
		int n= size(x);
		matrix[n, n] cov;
		cov = gp_exp_quad_cov(x, sdgp, lscale);
		for (i in 1:n)
			cov[i, i] = cov[i, i] + square(sigma);
		return cholesky_decompose(cov);
	}

	//additive kernel 2D
	matrix ker_gp1gp2(real[] x1, real[] x2, real sdgp1, real sdgp2, real lscale1, real lscale2, real sigma) { 
		matrix[size(x1), size(x1)] cov;
		cov = gp_exp_quad_cov(x1, sdgp1, lscale1) + gp_exp_quad_cov(x2, sdgp2, lscale2);
		for (n in 1:size(x1))
			cov[n, n] = cov[n, n] + square(sigma);
		return cholesky_decompose(cov);
	}
	
	//kernel 2D
	matrix ker_gp12(real[] x1, real[] x2, real sdgp, real lscale1, real lscale2, real sigma) { 
		matrix[size(x1), size(x1)] cov;
		cov = gp_exp_quad_cov(x1, sdgp, lscale1).*gp_exp_quad_cov(x2, 1, lscale2);
		for (n in 1:size(x1))
			cov[n, n] = cov[n, n] + square(sigma);
		return cholesky_decompose(cov);
	}
}

data {
	int N_pred;
	int<lower=1> D;
	vector[2] lscale;
	real gpscale;
	vector[D] x_pred[N_pred];
	vector[N_pred] eta;
}

transformed data{
}

parameters {
}

transformed parameters{
}

model{
}

generated quantities{
	vector[N_pred] f_true;
	f_true = ker_gp12(x_pred[,1], x_pred[,2], gpscale, lscale[1], lscale[2], 1e-5) * eta;
}


