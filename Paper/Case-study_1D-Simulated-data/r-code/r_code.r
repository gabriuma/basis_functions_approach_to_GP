
# CASE STUDY: 1D SIMULATED DATA

#------------------------------------------------------------------------------------------------
# Loading true function (GP prior with Mattern32 of marginal variance = 1 and lengthscale = 0.15)
#------------------------------------------------------------------------------------------------

load("f_true.rData")

#------------------------
# Input predictive space
#------------------------

x_pred <- seq(-1,1,0.002)		# vector of predictive input space
N_pred <- length(x_pred)		# length 
vv_pred <- 1:length(x_pred)		# indices 

#-----------------------------------
# Adding noise to the true fucntion
#-----------------------------------

seed <- 1234
set.seed(seed)

sd <- 0.2
y_pred <- f_true + rnorm(N_pred, 0, sd)

#--------------------------------------------------------
# A random sample (N=250) of the input predictive space
#--------------------------------------------------------

seed <- 1234
set.seed(seed)

vv <- sort(sample(vv_pred, 250))		# indices
x <- x_pred[vv]							# input vector
N <- length(x)							# vector length

#----------------------------
# Extrapolating test dataset
#----------------------------

vv_extra <- vv[which(x < (-0.8) | x > 0.8)]		# indices
N_extra <- length(vv_extra)						# length

#----------------------------
# Interpolating test dataset
#----------------------------

vv_inter <- vv[which((x > (-0.45) & x < (-0.36)) | (x > (-0.05) & x < 0.05) | (x > (0.45) & x < 0.6))]		# indices
N_inter <- length(vv_inter)		# length

#------------------
# Training dataset
#------------------

vv_train <- setdiff(vv, c(vv_extra, vv_inter))		# indices
N_train <- length(vv_train)							# length

#---------------------------------
# Plot training and test datasets
#---------------------------------

dev.new()
plot(x_pred[vv_train],y_pred[vv_train], cex=0.5, xlim=c(-1,1), ylim=c(-2.5,2.5))
points(x_pred[vv_extra],y_pred[vv_extra], col=2, cex=0.5)
points(x_pred[vv_inter],y_pred[vv_inter], col=2, cex=0.5)







