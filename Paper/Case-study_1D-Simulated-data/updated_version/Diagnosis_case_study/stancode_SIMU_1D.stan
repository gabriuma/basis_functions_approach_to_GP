functions {
	//GP 1D with a squared exponential (QE) kernel
	vector gp_QE(real[] x, real gpscale, real lscale, vector zgp) { 
		int n= size(x);
		matrix[n, n] cov;
		cov = cov_exp_quad(x, gpscale, lscale);
		for (i in 1:n)
			cov[i, i] = cov[i, i] + 1e-12;
		return cholesky_decompose(cov) * zgp;
	}
	//GP 1D with a Matern kernel
	vector gp_Mattern(real[] x, real gpscale, real lscale, vector zgp, int v) {
		int n= size(x);	
		matrix[n, n] cov;
		//Mattern 3/2
		if (v==3)
			for (i in 1:n)
				for (j in 1:n)
					cov[i,j] = gpscale^2 * (1 + (sqrt(3)*(fabs(x[i]-x[j])))/lscale) * exp(-sqrt(3)*(fabs(x[i]-x[j]))/lscale);
		//Mattern 5/2
		if (v==5)
			for (i in 1:n)
				for (j in 1:n)
					cov[i,j] = gpscale^2 * (1 + (sqrt(3)*(fabs(x[i]-x[j])))/lscale + (5*(fabs(x[i]-x[j]))^2)/(3*lscale^2)) * exp(-sqrt(5)*(fabs(x[i]-x[j]))/lscale);
		for (i in 1:n)
			cov[i, i] = cov[i, i] + 1e-12;
		return cholesky_decompose(cov) * zgp;
	}
}
data {
	int<lower=1> N;				
	real x[N];					
	int param_v;	//parameter v in the Mattern cov. fun.
	real lscale;
	real gpscale;
	vector[N] eta;
}
generated quantities{
	vector[N] f;
	f = gp_Mattern(x, gpscale, lscale, eta, param_v);
}
