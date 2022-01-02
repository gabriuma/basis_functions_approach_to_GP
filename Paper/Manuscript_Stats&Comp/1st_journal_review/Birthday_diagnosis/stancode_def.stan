functions {
  vector diagSPD_EQ(real gpscale, real lscale, real L, int M) {
    return sqrt((gpscale^2) * sqrt(2*pi()) * lscale * exp(-0.5*(lscale*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2));
  }
  vector diagSPD_periodic(real gpscale, real lscale, int M) {
    real a = 1/lscale^2;
    int one_to_M[M];
    for (m in 1:M) one_to_M[m] = m;
    vector[M] q = sqrt(gpscale^2 * 2 / exp(a) * to_vector(modified_bessel_first_kind(one_to_M, a)));
    return append_row(q,q);
  }
  matrix PHI_EQ(int N, int M, real L, vector x) {
    matrix[N,M] PHI = sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
	  for (m in 1:M)
       PHI[,m] = PHI[,m] - mean(PHI[,m]);
    return PHI;
  }
  matrix PHI_periodic(int N, int M, real w0, vector x) {
    matrix[N,M] mw0x = diag_post_multiply(rep_matrix(w0*x, M), linspaced_vector(M, 1, M));
    matrix[N,M] PHI = append_col(cos(mw0x), sin(mw0x));
    for (m in 1:M)
      PHI[,m] = PHI[,m] - mean(PHI[,m]);
    return PHI;
  }
}
data {
    real c_f1;                  //boundary value for function 1
    int<lower=1> M_f1;          //num basis functions for function 1
    int<lower=-1> J_f2;         //num cosine and sinu functions for function 3
    int<lower=-1> J_f3;         //num cosine and sinu functions for function 4
    int<lower=1> N;             //num observations
    vector[N] x;                //input vector
    vector[N] y;                //target vector
    real period_year;           //period of the year
    real period_week;           //period of the week
    int day_of_year2[N];        //day of the year inside a leap-year
}
transformed data {
    real L_f1= c_f1*max(fabs(x));
    //Basis functions for f1, f2 and f3
    matrix[N,M_f1] PHI_f1 = PHI_EQ(N, M_f1, L_f1, x);
    matrix[N,2*J_f2] PHI_f2 = PHI_periodic(N, J_f2, 2*pi()/period_year, x);
    matrix[N,2*J_f3] PHI_f3 = PHI_periodic(N, J_f3, 2*pi()/period_week, x);
    //
    //Simpler Horseshoe prior for special days effects (C.1 in Juho's article)
    // real<lower=0>  scale_global = 0.1;         //scale for the half-t prior for tau
    // real nu_global = 100;	   // degrees of freedom for the half-t priors for tau
    // real nu_local = 1;       // for the regularized horseshoe
    // real slab_scale = 2;     // for the regularized horseshoe
    // real slab_df = 100;      // for the regularized horseshoe
    //
    //More complex Horseshoe prior for special days effects (C.2 in Juho's article)
    // real<lower=0>  scale_global = 0.04;         //scale for the half-t prior for tau
    // real<lower=1>  nu_global = 1;             //df for the half-t priors for tau
    // real<lower=1>  nu_local = 1;              //df for the half-t priors for lambdas. (nu_local= 1 corresponds to the horseshoe)
}
parameters {
    real intercept;
    //variables for the basis function models
    vector[M_f1] beta_f1;
    vector[2*J_f2] beta_f2;
    vector[2*J_f3] beta_f3;
    //hyperparameters
    vector<lower=0>[3] lscale;
    vector<lower=0>[3] gpscale;
    real<lower=0> noise;
    //
    //Normal prior for special days effects
    // vector[366] f4;
    // real<lower=0> sigma_f4;
    //
    //t-student prior for special days effects
    vector[366] f4;
    real<lower=0> sigma_f4;
    //
    //Simpler Horseshoe prior for special days effects (C.1 in Juho's article)
    // vector[366] f4;
    // real<lower=0> tau_f4;           // global shrinkage parameter
    // vector<lower=0>[366] lambda_f4; // local shrinkage parameter
    // real<lower=0> caux_f4;           // auxiliary parameter
    //
    //More complex Horseshoe prior for special days effects (C.2 in Juho's article)
    // vector[366] z;
    // real<lower=0>  r1_global;
    // real<lower=0>  r2_global;
    // vector<lower=0>[366]  r1_local;
    // vector<lower=0>[366]  r2_local;
}
transformed parameters{
    vector[N] f;
    vector[N] f1;
    vector[N] f2;
    vector[N] f3;
    //
    //More complex Horseshoe prior for special days effects (C.2 in Juho's article)
    // vector[366] f4;
    {
    vector[M_f1] diagSPD_f1 = diagSPD_EQ(gpscale[1], lscale[1], L_f1, M_f1);
    vector[2*J_f2] diagSPD_f2 = diagSPD_periodic(gpscale[2], lscale[2], J_f2);
    vector[2*J_f3] diagSPD_f3 = diagSPD_periodic(gpscale[3], lscale[3], J_f3);
    //
    vector[M_f1] SPD_beta_f1 = diagSPD_f1 .* beta_f1;
    vector[2*J_f2] SPD_beta_f2 = diagSPD_f2 .* beta_f2;
    vector[2*J_f3] SPD_beta_f3 = diagSPD_f3 .* beta_f3;
    //
    f1 = PHI_f1[,] * SPD_beta_f1;
    f2 = PHI_f2[,] * SPD_beta_f2;
    f3 = PHI_f3[,] * SPD_beta_f3;
    f= intercept + f1 + f2 + f3 + f4[day_of_year2];
    //
    //More complex Horseshoe prior for special days effects (C.2 in Juho's article)
    // real tau = r1_global * sqrt(r2_global );            //global shrinkage parameter
    // vector[366] lambda_h = r1_local .* sqrt(r2_local);  //local shrinkage parameter
    // f4 = z .* lambda_h*tau;                             //function f4 (horseshoe effects)
    }
}
model{
    intercept ~ normal(0,1);
    beta_f1 ~ normal(0,1);
    beta_f2 ~ normal(0,1);
    beta_f3 ~ normal(0,1);
    //
    lscale ~ normal(0,2);           //GP lengthscales
    gpscale ~ normal(0,10);         //GP magnitudes
    noise ~ normal(0,1);            //model noise
    //
    //Normal prior for special days effects
    // f4 ~ normal(0, sigma_f4);
    // sigma_f4 ~ normal(0, 0.1);
    //
    //t-student prior for special days effects
    f4 ~ student_t(1, 0, sigma_f4);
    sigma_f4 ~ normal(0, 0.1);
    //
    //Simple Horseshoe parameterization for special days effects (C.1 in Juho's article)
    // real c_f4 = slab_scale * sqrt(caux_f4); // slab scale
    // f4 ~ normal(0, sqrt( c_f4^2 * square(lambda_f4) ./ (c_f4^2 + tau_f4^2*square(lambda_f4)))*tau_f4);
    // lambda_f4 ~ student_t(nu_local, 0, 1);
    // tau_f4 ~ student_t(nu_global, 0, scale_global*noise);
    // caux_f4 ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
    //
    //More complex Horseshoe prior for special days effects (C.2 in Juho's article)
    // z ~ normal(0, 1);
    // r1_local ~ normal (0.0,  1.0);
    // r2_local ~ inv_gamma (0.5* nu_local, 0.5* nu_local);
    // r1_global ~ normal (0.0,  scale_global*noise );
    // r2_global ~ inv_gamma (0.5* nu_global, 0.5* nu_global);
    //
    target += normal_lpdf(y | f, noise);
}
generated quantities{
  vector[N] y_rep;
  vector[N] log_lik;
  for(n in 1:N){
    y_rep[n] = normal_rng(f[n], noise);
    log_lik[n] = normal_lpdf(y[n] | f[n], noise);
  }
}