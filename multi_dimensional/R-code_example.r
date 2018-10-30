
library(viridis)
library(rstan)
library(rgl)
library(RColorBrewer)
library(brms)
library(rstanarm)
library(geoR)
library(lattice)
library(aqfig)
library(fields)
library(colorRamps)


#color palettes
col1 <- brewer.pal(11,"BrBG")[11:1] 
col2 <- cm.colors(11, alpha = 0.5)
col3 <- c(colorRampPalette(c("blue", "white"), bias=10)(15), colorRampPalette(c("white", "red"), bias=10)(15)[2:15])


# rho -> lengthscale
# alpha -> magnitud


#################################################
##### Application to the simulated dataset ######
#################################################

nobs <- 120  

f <- function(x1,x2) 1/100*(x1^3 - x2^3) + 1/10*x1*x2    #120obs 

x_grid <- pred_grid(coords= c(-5,5), y.coords= c(-5,5), by= 0.3)  #by=0.2 para function3

dev.new()
par(mfcol=c(2,2))

brk <- seq(min(f(x_grid[,1],x_grid[,2])),max(f(x_grid[,1],x_grid[,2])), length.out= 12)
xx <- unique(x_grid[,1])
image(xx, xx, matrix(f(x_grid[,1],x_grid[,2]),length(xx),length(xx)), col=col1, breaks= brk, main="f", xlab="x1", ylab="x2")

set.seed(10)  
x1 <- runif(nobs,-5,5)  
x2 <- runif(nobs,-5,5)
y <-  f(x1,x2) + rnorm(length(x1), 0, 0.3)

points(x1, x2, pch=1, cex=0.7)
plot(f(x1, x2), y, col=4, cex=0.6, xlab="f", ylab="y"); abline(a=0, b=1, col="gray") 

y_grid <- f(x_grid[,1],x_grid[,2]) + rnorm(length(dim(x_grid)[1]), 0, 0.2)

x <- rbind(cbind(x1,x2),as.matrix(x_grid))


########################
####  GP 2D
N1 <- length(x1)
Npred <- dim(x_grid)[1]
N <- N1 + Npred
D <- 2  #dimensions

standata_gp12 <- list(D= D, x= cbind(x1,x2), y= y, N1= N1, N= N, Npred= Npred, x_grid= x_grid, y_grid= y_grid)
str(standata_gp12)


stanout_gp12 <- stan(file= ".../basis_functions_approach_to_GP/multi_dimensional/simulated_dataset/model_GP_multi.stan", data= standata_gp12, iter= 300,  warmup= 200, chains= 1, thin= 1, algorithm= "NUTS")


summary(stanout_gp12, pars = c("sigma","rho","alpha"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary


gp12_f <- summary(stanout_gp12, pars = c("f"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
gp12_y <- summary(stanout_gp12, pars = c("y_predict"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
gp12_f_grid <- summary(stanout_gp12, pars = c("f_grid"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
gp12_y_grid <- summary(stanout_gp12, pars = c("y_grid_predict"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
gp12_elpd_grid <- summary(stanout_gp12, pars= c("log_y_grid_predict"), probs = c(0.025, 0.5, 0.975), digits_summary= 4)$summary
gp12_time <- sum(get_elapsed_time(stanout_gp12)[2])/(stanout_gp12@sim$iter - stanout_gp12@sim$warmup)


# Plots
dev.new()
par(mfcol=c(3,3), mai=c(0.4, 0.5, 0.25, 0.1), omi=c(0,0,0,0.2))

# training
plot(f(x1,x2), gp12_f[,1], col=4, cex=0.6, xlab="f", ylab="gp12_f", mgp= c(1.7, 0.5, 0), main="training", cex.main=0.7); abline(a=0, b=1, col="gray")
plot(f(x1,x2), f(x1,x2) - gp12_f[,1], xlab="f", ylab="f - gp12_f", col=4, cex=0.6, mgp= c(1.7, 0.5, 0), main="training", cex.main=0.7); abline(h=0, col="gray")

# predicting
plot(f(x_grid[,1],x_grid[,2]), gp12_f_grid[,1], col=4, cex=0.6, xlab="f", ylab="gp12_f", mgp= c(1.7, 0.5, 0), main="training + testing", cex.main=0.7); abline(a=0, b=1, col="gray")
plot(f(x_grid[,1],x_grid[,2]), f(x_grid[,1],x_grid[,2]) - gp12_f_grid[,1], xlab="f", ylab="f - gp12_f", col=4, cex=0.6, mgp= c(1.7, 0.5, 0), main="training + testing", cex.main=0.7); abline(h=0, col="gray")
plot(f(x_grid[,1],x_grid[,2]), gp12_elpd_grid[,1], col=2, cex=0.8, xlab="f", ylab="gp12_elpd", mgp= c(1.7, 0.5, 0), main="training + testing", cex.main=0.7); abline(h=mean(gp12_elpd_grid[,1]), col="red", lty=2)

# histogram residuals
gp12_error_grid <- matrix(f(x_grid[,1],x_grid[,2]),table(x_grid[,2])[1],table(x_grid[,2])[1]) - gp12_f_grid[,1]

hist(gp12_error_grid, mgp= c(1.7, 0.5, 0), xlab="f - gp12_f", ylab="", main="training + testing", cex.main=0.7)

#images
image(xx, xx, matrix(f(x_grid[,1],x_grid[,2]),length(xx),length(xx)), col=col1, breaks= brk, main="f", xlab="x1", ylab="x2", mgp= c(1.7, 0.5, 0))

image(xx, xx, matrix(gp12_f_grid[,1],length(xx),length(xx)), col= col1, breaks= brk , main="gp12_f", xlab="x1", ylab="x2", mgp= c(1.7, 0.5, 0))

brk1 <- seq(-max(abs(gp12_error_grid)),max(abs(gp12_error_grid)), length.out= 30)

image.plot(xx, xx, gp12_error_grid, col= col3, breaks= brk1, main="f - gp12_f", xlab="x1", ylab="x2", mgp= c(1.7, 0.5, 0), legend.width=1.2)
points(x1, x2, pch=1, cex=0.6)


##################################
##### Basis functions 2D

M <- c(2,4,6,10,15,20,30)  #number of basis functions
D <- 2  #number of dimensions

indices <- list()
for(j in 1:length(M)){
	indices[[j]] <- matrix(NA, M[j]^D, D)
	mm=0;
	for (m1 in 1:M[j]){
		for (m2 in 1:M[j]){
			mm = mm+1
			indices[[j]][mm,] = c(m1, m2)
		}
	}
}

standata_bf12 <- list()
for(j in 1:length(M)){
  	standata_bf12[[j]] <- list(D= D, M= M[j], M_nD= M[j]^D, L= c(5/2*max(x1),5/2*max(x2)), x= rbind(cbind(x1,x2),as.matrix(x_grid)), y= y, y_grid= y_grid, N1= N1, indices= indices[[j]], Npred= Npred)
}


stanout_bf12 <- list()
for(j in 1:length(M)){
	stanout_bf12[[j]] <- stan(file = ".../basis_functions_approach_to_GP/multi_dimensional/simulated_dataset/model_BF_multi.stan", data = standata_bf12[[j]], iter = 1000,  warmup = 500, chains=1, thin=1, algorithm = "NUTS", verbose = FALSE)
}


summary(stanout_bf12[[7]], pars = c("sigma","rho","alpha"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary


bf12_f <- list()
bf12_y <- list()
bf12_elpd <- list()
bf12_time <- list()
for(j in 1:length(M)){
	bf12_f[[j]] <- summary(stanout_bf12[[j]], pars = c("f"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
	bf12_y[[j]] <- summary(stanout_bf12[[j]], pars = c("y_predict"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
	bf12_elpd[[j]] <- summary(stanout_bf12[[j]], pars = c("log_y_predict"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
	bf12_time[[j]] <- sum(get_elapsed_time(stanout_bf12[[j]])[2])/(stanout_bf12[[j]]@sim$iter - stanout_bf12[[j]]@sim$warmup)
}

# Plots
for(j in 1:length(M)){
	dev.new()
	par(mfcol=c(3,3), mai=c(0.4, 0.5, 0.25, 0.1), omi=c(0,0,0,0.2))

	# training
	plot(gp12_f[,1], bf12_f[[j]][1:N1,1], col=4, cex=0.6, xlab="gp12_f ", ylab="bf12_f", mgp= c(1.7, 0.5, 0), main="training", cex.main=0.7); abline(a=0, b=1, col="gray")
	plot(gp12_f[,1], gp12_f[,1] - bf12_f[[j]][1:N1,1], xlab="gp12_f ", ylab="gp12_f - bf12_f", col=4, cex=0.6, mgp= c(1.7, 0.5, 0), main="training", cex.main=0.7); abline(h=0, col="gray")

	# predicting
	plot(gp12_f_grid[,1], bf12_f[[j]][-(1:N1),1], col=4, cex=0.6, xlab="gp12_f ", ylab="bf12_f", mgp= c(1.7, 0.5, 0), main="training + test", cex.main=0.7); abline(a=0, b=1, col="gray")
	plot(gp12_f_grid[,1], gp12_f_grid[,1] - bf12_f[[j]][-(1:N1),1], xlab="gp12_f ", ylab="gp12_f - bf12_f", col=4, cex=0.6, mgp= c(1.7, 0.5, 0), main="training + test", cex.main=0.7); abline(h=0, col="gray")
	plot(gp12_f_grid[,1], bf12_elpd[[j]][-(1:N1),1], col=2, cex=0.8, xlab="gp12_f ", ylab="bf12_elpd", mgp= c(1.7, 0.5, 0), main="training + test", cex.main=0.7); abline(h=mean(bf12_elpd[[j]][-(1:N1),1]), col="red", lty=2)
	
	# histogram residuals
	bf12_error <- matrix(gp12_f_grid[,1],table(x_grid[,2])[1],table(x_grid[,2])[1]) - bf12_f[[j]][-(1:N1),1]

	hist(bf12_error, mgp= c(1.7, 0.5, 0), xlab="gp12_f - bf12_f", ylab="", main="training + test", cex.main=0.7)
	
	#images
	image(xx, xx, matrix(gp12_f_grid[,1],length(xx),length(xx)), col=col1, breaks= brk, main="gp12_f", xlab="x1", ylab="x2", mgp= c(1.7, 0.5, 0))
	
	image(xx, xx, matrix(bf12_f[[j]][-(1:N1),1],length(xx),length(xx)), col= col1, breaks= brk, main="bf12_f", xlab="x1", ylab="x2", mgp= c(1.7, 0.5, 0))

	brk1 <- seq(-max(abs(bf12_error)),max(abs(bf12_error)), length.out= 30)
	
	image.plot(xx, xx, bf12_error, col= col3, breaks= brk1, main="gp12_f - bf12_f", xlab="x1", ylab="x2", mgp= c(1.7, 0.5, 0))
	points(x1, x2, pch=1, cex=0.5)
}


#################################################
##### Application to the Diabetes dataset #######
#################################################

# reference Notebook
# https://rawgit.com/avehtari/modelselection_tutorial/master/diabetes.html

diabetes <- read.csv(".../basis_functions_approach_to_GP/multi_dimensional/diabetes_dataset/diabetes.csv", header = TRUE)

summary(diabetes)
str(diabetes)

# removing those observation rows with 0 in any of the variables
for (i in 2:6) {
      diabetes <- diabetes[-which(diabetes[, i] == 0), ]
}
# scale the covariates for easier comparison of coefficient posteriors
for (i in 1:8) {
      diabetes[i] <- as.vector(scale(diabetes[i]))
}

# modify the data column names slightly for easier typing
names(diabetes)[7] <- "dpf"
names(diabetes) <- tolower(names(diabetes))

# bi-plots
p <- ggplot(diabetes, aes(age, glucose))
p + geom_point(aes(colour = outcome), size = 2)

dev.new()
p <- ggplot(diabetes, aes(pregnancies, glucose))
p + geom_point(aes(colour = outcome), size = 2)


### data

set.seed(115)
ind2 <- sample(1:dim(diabetes)[1], 15)
N2 <- length(ind2)

ind1 <- setdiff(1:dim(diabetes)[1], ind2)
N1 <- length(ind1)

x1 <- diabetes$pregnancies
x2 <- diabetes$glucose
x3 <- diabetes$age
y <- diabetes$outcome

length(x1)==N1+N2


##################
### GP 3D

standata_gp123 <- list()
standata_gp123$N1 <- N1
standata_gp123$N2 <- N2
standata_gp123$Npred <- 0
standata_gp123$ind1 <- ind1
standata_gp123$ind2 <- ind2
standata_gp123$D <- 3
standata_gp123$X <- data.frame(x1, x2, x3)
standata_gp123$Y <- y
standata_gp123$trials <- rep(1,N1+N2)


aj_gp123 <- stan(file = ".../basis_functions_approach_to_GP/multi_dimensional/diabetes_dataset/model_GP_multi.stan", data = standata_gp123, iter = 100,  warmup = 50, chains=1, thin=1, algorithm = "NUTS")


summary(aj_gp123, pars = c("rho","alpha"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary


gp123_f <- summary(aj_gp123, pars = c("f"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
gp123_y <- summary(aj_gp123, pars = c("y_predict"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
gp123_finv <- summary(aj_gp123, pars = c("f_invlogit"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
gp123_elpd <- summary(aj_gp123, pars = c("log_y_predict"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
gp123_time <- sum(get_elapsed_time(aj_gp123)[2])/(aj_gp123@sim$iter - aj_gp123@sim$warmup)


# Plots
dev.new()
par(mfrow=c(2,2))

hist(as.numeric(gp123_y[,1]>0.5), main="", xlab="gp123_y")
hist(y, main="", xlab="y")

plot(gp123_y[ind2,1], gp123_elpd[ind2,1], col=4, cex=0.6, xlab="gp12_y", ylab="gp12_elpd"); abline(a=0, b=1, col="gray")

table(as.factor(as.numeric(gp123_y[,1]>0.5)), y)

table(as.factor(as.numeric(gp123_y[ind2,1]>0.5)), y[ind2])


####################
####  BF 3D

D <- 3
M <- c(2,4,6,10)  #,20,30,40,60) 

indices <- list()
for(j in 1:length(M)){
	indices[[j]] <- matrix(NA, M[j]^D, D)
	mm=0;
	for (m1 in 1:M[j]){
		for (m2 in 1:M[j]){
			for (m3 in 1:M[j]){
				mm = mm+1
				indices[[j]][mm,] = c(m1, m2, m3)
			}
		}
	}
}

standata_bf123 <- list()
for(j in 1:length(M)){
  	standata_bf123[[j]] <- standata_gp123
	standata_bf123[[j]]$D <- D
	standata_bf123[[j]]$M <- M[j]
	standata_bf123[[j]]$M_nD <- M[j]^D
	standata_bf123[[j]]$indices <- indices[[j]]
	standata_bf123[[j]]$L <- c(5/2*max(x1),5/2*max(x2),5/2*max(x3))
}


stanout_bf123 <- list()
for(j in 1:length(M)){
	stanout_bf123[[j]] <- stan(file = ".../basis_functions_approach_to_GP/multi_dimensional/diabetes_dataset/model_BF_multi.stan", data = standata_bf123[[j]], iter = 1000,  warmup = 500, chains=1, thin=1, algorithm = "NUTS")
}


summary(stanout_bf123[[2]], pars = c("rho","alpha"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary


bf123_f <- list()
bf123_finv <- list()
bf123_y <- list()
bf123_elpd <- list()
bf123_time <- list()
for(j in 1:length(M)){
	bf123_f[[j]] <- summary(stanout_bf123[[j]], pars = c("f"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
	bf123_finv[[j]] <- summary(stanout_bf123[[j]], pars = c("f_invlogit"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
	bf123_y[[j]] <- summary(stanout_bf123[[j]], pars = c("y_predict"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
	bf123_elpd[[j]] <- summary(stanout_bf123[[j]], pars = c("log_y_predict"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
	bf123_time[[j]] <- sum(get_elapsed_time(stanout_bf123[[j]])[2])/(stanout_bf123[[j]]@sim$iter - stanout_bf123[[j]]@sim$warmup)
}

# Plots
for(j in 1:length(M)){
	dev.new()
	par(mfrow=c(2,4))

	plot(gp123_f[,1], bf123_f[[j]][,1], col=4, cex=0.6, xlab="gp123_f", ylab="bf123_f"); abline(a=0, b=1, col="gray")

	plot(gp123_f[,1], gp123_f[,1] - bf123_f[[j]][,1], col=4, cex=0.6, xlab="gp123_f", ylab="gp123_f - bf123_f"); abline(h=0, col="gray")
	
	plot(gp123_finv[,1], bf123_finv[[j]][,1], col=4, cex=0.6, xlab="gp123_finv", ylab="bf123_finv"); abline(a=0, b=1, col="gray")

	plot(gp123_finv[,1], gp123_finv[,1] - bf123_finv[[j]][,1], col=4, cex=0.6, xlab="gp123_finv", ylab="gp123_finv - bf123_finv"); abline(h=0, col="gray")
	
	plot(gp123_y[,1], bf123_y[[j]][,1], col=4, cex=0.6, xlab="gp123_y", ylab="bf123_y"); abline(a=0, b=1, col="gray")

	plot(gp123_y[,1], gp123_y[,1] - bf123_y[[j]][,1], col=4, cex=0.6, xlab="gp123_y", ylab="gp123_y - bf123_y"); abline(h=0, col="gray")
	
	hist(as.numeric(bf123_y[[j]][,1]>0.5))
	hist(y)
	
	print(table(as.factor(as.numeric(bf123_y[[j]][,1]>0.5)), y))

	print(table(as.factor(as.numeric(bf123_y[[j]][ind2,1]>0.5)), y[ind2]))
}


