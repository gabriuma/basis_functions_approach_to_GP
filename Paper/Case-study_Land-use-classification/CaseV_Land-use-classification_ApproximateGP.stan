functions{
	real lambda(real L, int m) {
		real lam;
		lam = ((m*pi())/(2*L))^2;
				
		return lam;
	}
	real spd(real alpha, real rho, real w) {
		real S;
		S = (alpha^2) * sqrt(2*pi()) * rho * exp(-0.5*(rho^2)*(w^2));
				
		return S;
	}
	vector phi(real L, int m, vector x) {
		vector[rows(x)] fi;
		fi = 1/sqrt(L) * sin(m*pi()/(2*L) * (x+L));
				
		return fi;
	}
	vector SPD(real L, real alpha, real rho, int M) {
		vector[M] diagS;
		for(m in 1:(M)){ diagS[m] = sqrt(spd(alpha, rho, sqrt(lambda(L, m)))); }
				
		return diagS;
	}
	vector SPD_beta(real L, real alpha, real rho, vector beta, int M) {
		vector[M] diagS_beta;
		diagS_beta = SPD(L, alpha, rho, M) .* beta[1:M];
				
		return diagS_beta;
	}
	vector fx(real L, real alpha, real rho, matrix PHI, vector beta, int M) {
		vector[rows(PHI)] fx;
		fx= PHI[,1:M] * SPD_beta(L, alpha, rho, beta, M);
				
		return fx;
	}
	
	vector lambda_nD(real[] L, int[] m, int D) {
		vector[D] lam;
		for(i in 1:D){
			lam[i] = ((m[i]*pi())/(2*L[i]))^2; }
				
		return lam;
	}
	real spd_nD(real alpha, row_vector rho, vector w, int D) {
		real S;
		S = alpha^2 * sqrt(2*pi())^D * prod(rho) * exp(-0.5*((rho .* rho) * (w .* w)));
				
		return S;
	}
	vector phi_nD(real[] L, int[] m, matrix x) {
		int c = cols(x);
		int r = rows(x);
		
		matrix[r,c] fi;
		vector[r] fi1;
		for (i in 1:c){
			fi[,i] = 1/sqrt(L[i])*sin(m[i]*pi()*(x[,i]+L[i])/(2*L[i]));
		}
		fi1 = fi[,1];
		for (i in 2:c){
			fi1 = fi1 .* fi[,i];
		}
		return fi1;
	}
	vector SPD_nD(real[] L, real alpha, row_vector rho, int[,] indices, int D, int M_nD) {
		vector[M_nD] diagS;
		for(m in 1:M_nD){ diagS[m] = sqrt(spd_nD(alpha, rho, sqrt(lambda_nD(L, indices[m,], D)), D)); }
				
		return diagS;
	}
	vector SPD_beta_nD(real[] L, real alpha, row_vector rho, int[,] indices, int D, int M_nD, vector beta) {
		vector[M_nD] diagS_beta;
		diagS_beta = SPD_nD(L, alpha, rho, indices, D, M_nD) .* beta[1:M_nD];
				
		return diagS_beta;
	}
	vector fx_nD(real[] L, matrix PHI, int[,] indices, real alpha, row_vector rho, int D, int M_nD, vector beta) {
		vector[rows(PHI)] fx_nD;
		fx_nD= PHI[,1:M_nD] * SPD_beta_nD(L, alpha, rho, indices, D, M_nD, beta);
				
		return fx_nD;
	}
}

data { 
	int<lower=2> K; 
	int<lower=1> N; 
	int<lower=1> D; 
	matrix[N,D] x;
	
	int<lower=1,upper=K> y[N];
	int<lower=1> N1;
	int<lower=0> N2;
	int<lower=0> Npred;
	int ind1[N1];
	int ind2[N2];

	real L[D];

	int<lower=1> M[D];			
	int M_nD[D-1,D];

	int<lower=0> max_M_nD;	
	int indices[D-1,D,max_M_nD,2];
} 

transformed data { 
	matrix[max(M),N] FI[D];
	matrix[max_M_nD,N] FI2[D-1,D];
	int max_M = max(M);
	int Dt= D;
	int Dc= 4;
	
	for(i in 1:D){ 
		for(m in 1:M[i]){ 
			FI[i,m,ind1] = to_row_vector(phi(L[i], m, x[ind1,i])); 
			FI[i,m,ind2] = to_row_vector(phi(L[i], m, x[ind2,i]));
		}
		
		if(M[i]!=max_M) { 
			for(m in (M[i]+1):max_M) { FI[i,m] = rep_row_vector(-9999, N); }
		}

	}
	
	for(i in 1:(D-1)){ 
		for(j in (i+1):D){
			for(m in 1:M_nD[i,j]){ 
				if((i==1 && j==Dt)  || (i==2 && j==Dt) || (i==3 && j==Dt) || (i==4 && j==Dt) //#|| (i==5 && j==Dt) || (i==6 && j==Dt) //# || (i==7 && j==Dt) || (i==8 && j==Dt) //# || (i==9 && j==Dt) || (i==10 && j==Dt) || (i==11 && j==Dt) || (i==12 && j==Dt) || (i==13 && j==Dt)
				){
					FI2[i,j,m,ind1] = to_row_vector( phi_nD({L[i],L[j]}, indices[i,j,m], x[ind1,{i,j}]) );
					FI2[i,j,m,ind2] = to_row_vector( phi_nD({L[i],L[j]}, indices[i,j,m], x[ind2,{i,j}]) );				
				}
				else{
					FI2[i,j,m,ind1] = rep_row_vector(9999,N1);
					FI2[i,j,m,ind2] = rep_row_vector(9999,N2);
				}
			}
			
			if(M_nD[i,j]!=max_M_nD) { 
				for(m in (M_nD[i,j]+1):max_M_nD) { FI2[i,j,m] = rep_row_vector(-9999, N); }
			}
		}
	}

}

parameters { 
	row_vector<lower=0,upper=5>[D] rho;
	matrix<lower=0,upper=5>[Dc,1] rho2;
	
	matrix<lower=0>[D,K-1] alpha_1;
	matrix<lower=0>[Dc,1] alpha_2[K-1];
	
	vector[max_M] beta[D,K-1];
	vector[max_M_nD] beta2[Dc,1,K-1];
}

transformed parameters { 
	vector[K] theta[N1];
	matrix[N1,K] f;
	
	matrix[N1,K-1] f_raw;
	vector[N1] f_raw1[D,K-1];
	
	vector[N1] f_raw2[Dc,1,K-1];

	//#//#INDIVIDUAL EFFECTS
	for(i in 1:D){
		for(k in 1:(K-1)){
			if(i<=D){
				f_raw1[i,k] = fx(L[i], alpha_1[i,k], rho[i], FI[i,,ind1]', beta[i,k], M[i]); //#' 
			}else{
				f_raw1[i,k] = rep_vector(9999,N1);
			}
			
			//#f_raw2[i,i,k] = f_raw1[i,k];
		}
	}
	
	for(k in 1:(K-1)){
		
		//#//#//#PAIR-WISE INTERACTIONS
		for(i in 1:Dc){ //#
			for(j in 1:1){
				if((i==1 && j==1) || (i==2 && j==1) || (i==3 && j==1) || (i==4 && j==1) //#|| (i==5 && j==1) || (i==6 && j==1) //#|| (i==7 && j==1) || (i==8 && j==1) //#|| (i==9 && j==1) || (i==10 && j==1) || (i==11 && j==1) || (i==12 && j==1) || (i==13 && j==1)
				){
					f_raw2[i,j,k] = fx_nD({L[i],L[Dt]}, FI2[i,Dt][,ind1]', indices[i,Dt], alpha_2[k,i,1], rep_row_vector(rho2[i,1],2), 2, M_nD[i,Dt], beta2[i,1,k]); //#'  //#rho[{i,Dt}]
				}else{
					f_raw2[i,j,k] = rep_vector(9999,N1);
				}
				//# f_raw2[j,i,k] = f_raw2[i,j,k];
			}
		}
		
		f_raw[,k] = f_raw1[1,k] + f_raw1[2,k] + f_raw1[3,k] + f_raw1[4,k] + f_raw1[5,k] + f_raw1[6,k] + f_raw1[7,k] + f_raw1[8,k] + f_raw1[9,k] + f_raw1[10,k] + f_raw1[11,k] + f_raw1[12,k] + f_raw1[13,k] + f_raw1[14,k] + f_raw1[15,k] + f_raw1[16,k] + f_raw1[17,k] + f_raw1[18,k] + f_raw1[19,k] + f_raw1[20,k] + f_raw1[21,k] + f_raw1[22,k] + f_raw1[23,k] + f_raw1[24,k] + f_raw1[25,k] + f_raw1[26,k] + f_raw1[27,k] + f_raw1[28,k] + f_raw1[29,k] + f_raw1[30,k] + f_raw1[31,k] + f_raw1[32,k] + f_raw1[33,k] + f_raw1[34,k] + f_raw1[35,k] + f_raw1[36,k]+ f_raw1[37,k]+ f_raw1[38,k]+ f_raw1[39,k] + f_raw1[40,k]+ f_raw1[41,k]+ f_raw1[42,k]+ f_raw1[43,k]+ f_raw1[44,k]+ f_raw1[45,k]+ f_raw1[46,k]+ f_raw1[47,k]+ f_raw1[48,k]+ f_raw1[49,k]+ f_raw1[50,k]+ f_raw1[51,k] + f_raw1[52,k] + f_raw1[53,k] + f_raw1[54,k] + f_raw2[1,1,k] + f_raw2[2,1,k] + f_raw2[3,1,k] + f_raw2[4,1,k]; //# + f_raw2[5,1,k] + f_raw2[6,1,k] ;//# + f_raw2[7,1,k] + f_raw2[8,1,k]; //# + f_raw2[9,1,k] + f_raw2[10,1,k] + f_raw2[11,1,k] + f_raw2[12,1,k] + f_raw2[13,1,k]; 
		 

		f[,k] = f_raw[,k];
	}
	for(i in 1:N1){
		f[i,K] = -sum(f_raw[i,]);
	}
	for (i in 1:N1){
		theta[i] = softmax(f[i,]'); //#'
	}
}

model { 
	for(k in 1:(K-1)){
		for(i in 1:D){
			beta[i,k] ~ normal(0,1);
		}
		for(i in 1:Dc){
			for(j in 1:1){
				beta2[i,j,k] ~ normal(0,1);
			}
		}
	}

	rho ~ inv_gamma(2,4);
	to_vector(rho2) ~ inv_gamma(2,4);
	
	to_vector(alpha_1) ~ inv_gamma(2,5);
	
	for(k in 1:(K-1)){
		to_vector(alpha_2[k]) ~ inv_gamma(2,5);
	}
	
	for (i in 1:N1)	y[ind1[i]] ~  categorical(theta[i,]);
}

generated quantities{
	vector[K] theta_p[N2];
	matrix[N2,K] f_p;
	
	matrix[N2,K-1] f_raw_p;
	
	vector[N2] f_raw1_p[D,K-1];
	vector[N2] f_raw2_p[Dc,1,K-1];

	
	for(i in 1:D){
		for(k in 1:(K-1)){
		
			//#INDIVIDUAL EFFECTS
			f_raw1_p[i,k] = fx(L[i], alpha_1[i,k], rho[i], FI[i][,ind2]', beta[i,k], M[i]); //#'
			
			//#PAIR-WISE INTERACTIONS
			//# f_raw2_p[i,i,k] = rep_vector(9999,N2);//# f_raw1_p[i,k];
		}
	}
	
	for(k in 1:(K-1)){
	
		//# PAIR-WISE INTERACTIONS
		for(i in 1:Dc){  //#
			for(j in 1:1){
				if((i==1 && j==1) || (i==2 && j==1) || (i==3 && j==1) || (i==4 && j==1) //# || (i==5 && j==1) || (i==6 && j==1) //#|| (i==7 && j==1) || (i==8 && j==1) //# || (i==9 && j==1) || (i==10 && j==1) || (i==11 && j==1) || (i==12 && j==1) || (i==13 && j==1)
				){
					f_raw2_p[i,j,k] = fx_nD({L[i],L[Dt]}, FI2[i,Dt][,ind2]', indices[i,Dt], alpha_2[k,i,1], rep_row_vector(rho2[i,1],2), 2, M_nD[i,Dt], beta2[i,1,k]); //#' 
				}else{
					f_raw2_p[i,j,k] = rep_vector(9999,N2);
				}
				//# f_raw2_p[j,i,k] = f_raw2_p[i,j,k];
			}
		}
		
		f_raw_p[,k] = f_raw1_p[1,k] + f_raw1_p[2,k] + f_raw1_p[3,k] + f_raw1_p[4,k] + f_raw1_p[5,k] + f_raw1_p[6,k] + f_raw1_p[7,k] + f_raw1_p[8,k] + f_raw1_p[9,k] + f_raw1_p[10,k] + f_raw1_p[11,k] + f_raw1_p[12,k] + f_raw1_p[13,k] + f_raw1_p[14,k] + f_raw1_p[15,k] + f_raw1_p[16,k] + f_raw1_p[17,k] + f_raw1_p[18,k] + f_raw1_p[19,k] + f_raw1_p[20,k] + f_raw1_p[21,k] + f_raw1_p[22,k] + f_raw1_p[23,k] + f_raw1_p[24,k] + f_raw1_p[25,k] + f_raw1_p[26,k] + f_raw1_p[27,k] + f_raw1_p[28,k] + f_raw1_p[29,k] + f_raw1_p[30,k] + f_raw1_p[31,k] + f_raw1_p[32,k] + f_raw1_p[33,k] + f_raw1_p[34,k] + f_raw1_p[35,k] + f_raw1_p[36,k]+ f_raw1_p[37,k]+ f_raw1_p[38,k]+ f_raw1_p[39,k] + f_raw1_p[40,k]+ f_raw1_p[41,k]+ f_raw1_p[42,k]+ f_raw1_p[43,k]+ f_raw1_p[44,k]+ f_raw1_p[45,k]+ f_raw1_p[46,k]+ f_raw1_p[47,k]+ f_raw1_p[48,k]+ f_raw1_p[49,k]+ f_raw1_p[50,k] + f_raw1_p[51,k] + f_raw1_p[52,k] + f_raw1_p[53,k] + f_raw1_p[54,k] + f_raw2_p[1,1,k] + f_raw2_p[2,1,k] + f_raw2_p[3,1,k] + f_raw2_p[4,1,k]; //# + f_raw2_p[5,1,k] + f_raw2_p[6,1,k] ;//#+ f_raw2_p[7,1,k]+ f_raw2_p[8,1,k];
		//#  + f_raw2_p[9,1,k] + f_raw2_p[10,1,k] + f_raw2_p[11,1,k] + f_raw2_p[12,1,k] + f_raw2_p[13,1,k];//# 


		f_p[,k] = f_raw_p[,k];
	}
	for(i in 1:N2){
		f_p[i,K] = -sum(f_raw_p[i,]);
	}
	for (i in 1:N2){
		theta_p[i] = softmax(f_p[i,]'); //#'
	}
}

