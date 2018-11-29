
library(rstan)

data <- read.csv(file="C:/GABRIEL_20180206/GIFLE/Proyecto-GAMs/Basisfunctions/BF-1D/Birthday_project/births_usa_1969.csv")
str(data)

# input
x <- scale(data$id[], center=TRUE, scale=TRUE)

# response variable
y <- scale(data$births[], center=TRUE, scale=TRUE)

# std and mean of the response variable
std_y <- attr(y,"scaled:scale")
m_y <- attr(y,"scaled:center")

# std of the input
std_x <- attr(x,"scaled:scale")

# period for year and week
period_year <- 365.25/std_x
period_week <- 7/std_x


M_f1 <- 30		#num basis functions for f1= smoth trend
c_f1 <- 1.5		#factor c for f1= smoth trend
J_f3 <- 6		#num basis functions for f3= year effect
J_f4 <- 4		#num basis functions for f4= week effect

standata <- list(M_f1= M_f1, L_f1= c_f1*max(abs(x)), J_f3= J_f3, J_f4= J_f4, x= x[,1], y= y[,1], N= length(x), period_year= period_year, period_week= period_week) 
str(standata)


# Run Stan
stanout <- stan(file = "C:/GABRIEL_20180206/GIFLE/Proyecto-GAMs/Basisfunctions/BF-1D/Birthday_project/stancode_v3.stan", data= standata, iter= 50,  warmup= 20, chains= 1, thin= 1, algorithm= "NUTS", verbose= FALSE, control=list(adapt_delta =0.99, max_treedepth= 15))


# Load the output
load("C:/GABRIEL_20180206/GIFLE/Proyecto-GAMs/Basisfunctions/BF-1D/Birthday_project/stanout.rData")
ls()


#Storing the results
f <- summary(stanout, pars = c("f"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
f1 <- summary(stanout, pars = c("f1"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
f3 <- summary(stanout, pars = c("f3"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
f4 <- summary(stanout, pars = c("f4"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
f5 <- summary(stanout, pars = c("f5"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
rho <- summary(stanout, pars = c("rho"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
alpha <- summary(stanout, pars = c("alpha"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
sigma <- summary(stanout, pars = c("sigma"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
tau <- summary(stanout, pars = c("tau"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
lambda_h <- summary(stanout, pars = c("lambda_h"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary
z <- summary(stanout, pars = c("z"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary


#Printing the results
print(rho[,c(1,3,7,8)])
print(alpha[,c(1,3,7,8)])
print(sigma[,c(1,3,7,8)])

print(head(f[,c(1,3,7,8)]))
print(head(f1[,c(1,3,7,8)]))
print(head(f3[,c(1,3,7,8)]))
print(head(f4[,c(1,3,7,8)]))
print(head(f5[,c(1,3,7,8)]))

print(tau[,c(1,3,7,8)])
print(head(lambda_h[,c(1,3,7,8)]))
print(head(z[,c(1,3,7,8)]))


### PLOT OF ONLY ONE YEAR
dev.new()

data_year <- data[data$year==1972,]
ind <- data_year$id
axis_labels_at <- aggregate(data_year, by=list(data_year$month), FUN=min)$id

plot(ind, (y[ind]*std_y+m_y)/m_y, type="p", lty=1, pch=18, cex=0.4, col="black", xlab= "", ylab= "births/mean_births", cex.lab=1.5, cex.axis=1.5, xaxt="n", ylim=c(0.7,1.3), main=paste(data_year$year[1])) 

axis(1, at= axis_labels_at, labels= c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"), tick = TRUE, cex.axis=1.5)

lines(ind, (f[ind,1]*std_y+m_y)/m_y, col="black", lwd=1)	# f
lines(range(ind), c(1,1), lty=2)							# mean
lines(ind, (f1[ind,1]*std_y+m_y)/m_y, col=2, lwd=2)			# f1 smoth trend
lines(ind, (f3[ind,1]*std_y+m_y)/m_y, col=3, lwd=2)	 		# f3 year effect
lines(ind, (f5[ind,1]*std_y+m_y)/m_y, col=6, lwd=2) 		# f5 horseshoe

#labels special days
abline(v=ind[data_year$month==2&data_year$day==14], lty=2, col="grey")
text(ind[data_year$month==2&data_year$day==14], y = 1.1, labels = "Valentine's day", pos =NULL, offset = 0)
abline(v= ind[data_year$month==2&data_year$day==29], lty=2, col="grey")
text(ind[data_year$month==2&data_year$day==29], y = 1.08, labels = "Leap day", pos =NULL, offset = 0)
abline(v= ind[data_year$day==13], lty=2, col="grey")
text(ind[data_year$day==13], y = 1.12, labels = "13th", pos =NULL, offset = 0)

legend("topleft",legend=c("Observations","Mean","Smooth trend (S)","Year effect (Y)","Special days effect (E)","S + Y + E + Week effect"), col=c(1,1,2,3,6,"black"), lty=c(NA,2,1,1,1,1), pch=c(18,NA,NA,NA,NA,NA), lwd=c(1,2,3,3,3,3), cex=1.3)


### PLOT OF ALL THE YEARS
dev.new()

str(data[,])
ind <- data$id

labels_at = aggregate(data, by=list(data$year), FUN=min)$id

plot(ind, (y[ind]*std_y+m_y)/m_y, type="p", lty=1, pch=18, cex=0.4, col="black", xlab= "", ylab= "births/mean_births",cex.lab=1.5, cex.axis=1.5, xaxt="n", ylim=c(0.7,1.3))

axis(1, at = labels_at, labels = c("1969","1970","1971","1972","1973","1974","1975","1976","1977","1978","1979","1980","1981","1982","1983","1984","1985","1986","1987","1988"), tick = TRUE, cex.axis=1.5)

lines(ind, (f[ind,1]*std_y+m_y)/m_y, col="grey", lwd=1)			# f
lines(c(0,7305), c(1,1), lty=2, lwd=1)							# mean
lines(ind, (f1[ind,1]*std_y+m_y)/m_y, col=2, lwd=1)				# smooth trend
lines(ind, (f3[ind,1]*std_y+m_y)/m_y, col=3, lwd=1)				# year effect
# lines(ind1[], (f5_BF[[1]][,1]*sd_y+m_y)/m_y, col=6, lwd=1)	# horseshoe

legend(x=3*365, y=1.32,legend=c("Observations","Mean","Smooth trend (S)","Year effect (Y)","S + Y + Special days effect + Week effect"), col=c(1,1,2,3,"grey"), lty=c(NA,2,1,1,1), pch=c(18,NA,NA,NA,NA), lwd=c(1,2,3,3,3), cex=1.3)


### PLOT THE WEEK EFFECT
dev.new()

str(data[,])
ind <- data$id

plot(ind[6:12], (f4[6:12,1]*std_y+m_y)/m_y, type="b", lty=1, pch=1, cex=1, col=4, xlab= "", ylab= "births/mean_births",cex.lab=1.5, cex.axis=1.5, xaxt="n", ylim=c(0.7,1.3))

axis(1, at = c(6:12), labels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun"), tick = TRUE)
abline(h=1, lty=2)

legend("topleft",legend=c("Week effect (W)"), col=c(4), lty=c(1), lwd=c(2), cex= 1.3)


### PLOT THE FIRST FOUR YEARS
dev.new()

data_year <- data[data$year==1969 | data$year==1970 | data$year==1971 | data$year==1972,]
ind <- data_year$id
axis_labels_at <- aggregate(data_year, by=list(data_year$year), FUN=min)$id

plot(ind, (y[ind]*std_y+m_y)/m_y, type="p", lty=1, pch=18, cex=0.4, col="black", xlab= "", ylab= "births/mean_births",cex.lab=1.5, cex.axis=1.5, xaxt="n", ylim=c(0.7,1.3))

axis(1, at = axis_labels_at, labels = c("1969","1970","1971","1972"), tick = TRUE, cex.axis=1.5)

lines(ind, (f[ind,1]*std_y+m_y)/m_y, col="grey", lwd=1)		# f
lines(range(ind), c(1,1), lty=2, lwd=2)						# mean
lines(ind, (f1[ind,1]*std_y+m_y)/m_y, col=2, lwd=1)			# f1
lines(ind, (f3[ind,1]*std_y+m_y)/m_y, col=3, lwd=1)	 		# f3
lines(ind, (f5[ind,1]*std_y+m_y)/m_y, col=6, lwd=1) 		# f5

legend("topleft",legend=c("Observations","Mean","Smooth trend (S)","Year effect (Y)","Special days effect (E)","S + Y + E + Week effect"), col=c(1,1,2,3,6,"grey"), lty=c(NA,2,1,1,1,1), pch=c(18,NA,NA,NA,NA,NA), lwd=c(1,2,3,3,3,3), cex=1.3)





