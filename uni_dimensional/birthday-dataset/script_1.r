
library(rstan)
library(latex2exp)

data <- read.csv(file="C:/GABRIEL_20192301/Proyecto-GAMs/Basisfunctions/BF-1D/Birthday_project/births_usa_1969.csv")
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
load("C:/GABRIEL_20192301/Proyecto-GAMs/Basisfunctions/BF-1D/Birthday_project/stanout_100iter.rData")
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
par(mai=c(1.02, 1.10, 0.82, 0.42))

data_year <- data[data$year==1972,]
ind <- data_year$id
axis_labels_at <- aggregate(data_year, by=list(data_year$month), FUN=min)$id

plot(ind, (y[ind]*std_y+m_y)/m_y, type="p", pch=21, bg=grey(0.7), cex=0.9, col=grey(0.4), xlab="", ylab="", lwd=1, ylim=c(0.7,1.2), mgp= c(3.5, 1, 0), frame.plot = TRUE, yaxs="r", cex.lab=2.5, las=1, xaxt="n", yaxt="n",fg=grey(0.5), family="serif")
axis(1, at = axis_labels_at, labels = c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"), tick = TRUE, lty = 1, mgp= c(3, 1.4, 0), las=1, cex.axis=2.5, font=1, col=grey(0.5), col.ticks=grey(0.3), family="")
axis(2, at = NULL, labels = TRUE, tick = TRUE, lty = 1, mgp= c(3, 0.7, 0), las=1, cex.axis=2.5, font=5, col=grey(0.5), col.ticks=grey(0.3)) #mgp= c(2.5, 0.7, 0)
title(xlab ="Month", mgp= c(3.7, 1, 0), cex.lab=2, las=1)
title(ylab ="Proportion of births over the mean", mgp= c(4.1, 0.7, 0), cex.lab=2, las=1)
	
lines(ind, (f[ind,1]*std_y+m_y)/m_y, col="black", lwd=1)	# f
# lines(range(ind), c(1,1), lty=2)
abline(h=1, lty=2)						# mean
lines(ind, (f1[ind,1]*std_y+m_y)/m_y, col=2, lwd=2)			# f1 smoth trend
lines(ind, (f3[ind,1]*std_y+m_y)/m_y, col=3, lwd=2)	 		# f3 year effect
lines(ind, (f5[ind,1]*std_y+m_y)/m_y, col=6, lwd=2) 		# f5 horseshoe

#labels special days
text(ind[data_year$month==1&data_year$day==1], y = 0.85, labels = "New year", pos =NULL, offset = 0, family="serif", cex=1.3)
text(ind[data_year$month==2&data_year$day==14], y = 1.04, labels = "Valentine's day", pos =NULL, offset = 0, family="serif", cex=1.3)
text(ind[data_year$month==2&data_year$day==29], y = 0.98, labels = "Leap day", pos =NULL, offset = 0, family="serif", cex=1.3)
text(ind[data_year$month==4&data_year$day==1], y = 0.985, labels = "April 1st", pos =NULL, offset = 0, family="serif", cex=1.3)
text(ind[data_year$month==5&data_year$day==27], y = 0.98, labels = "Memorial day", pos =NULL, offset = 0, family="serif", cex=1.3)
text(ind[data_year$month==7&data_year$day==4], y = 0.86, labels = "Independence day", pos =NULL, offset = 0, family="serif", cex=1.3)
text(ind[data_year$month==9&data_year$day==2], y = 0.94, labels = "Labor day", pos =NULL, offset = 0, family="serif", cex=1.3)
text(ind[data_year$month==10&data_year$day==30], y = 0.99, labels = "Halloween", pos =NULL, offset = 0, family="serif", cex=1.3)
text(ind[data_year$month==11&data_year$day==25], y = 0.94, labels = "Thanks-giving", pos =NULL, offset = 0, family="serif", cex=1.3)
text(ind[data_year$month==12&data_year$day==25], y = 0.82, labels = "Christmas", pos =NULL, offset = 0, family="serif", cex=1.3)

# abline(v= ind[data_year$day==13], lty=2, col="grey")

legend("topleft",inset=c(0.1,0.05),legend=c("Observations",TeX('Long-term trend ($f_1$)'),TeX('Year effects ($f_2$)'),TeX('Special days effects ($f_4$)'),TeX('$\\mu$=f_1+f_2+f_3+f_4$')), col=c(grey(0.4),2,3,6,"black"), lty=c(NA,1,1,1,1), pch=c(20,NA,NA,NA,NA), lwd=c(2,3,3,3,3), cex=1.7, xpd=TRUE, bty="n", y.intersp=1, x.intersp=0.8, text.font=1, ncol=2, seg.len=1.3)


### PLOT OF ALL THE YEARS
dev.new()
par(mai=c(1.02, 1.10, 0.82, 0.42))

str(data[,])
ind <- data$id

labels_at = aggregate(data, by=list(data$year), FUN=min)$id

plot(ind, (y[ind]*std_y+m_y)/m_y, type="p", pch=20, bg=grey(0.4), cex=0.6, col=grey(0.5), xlab="", ylab="", lwd=1, ylim=c(0.7,1.3), mgp= c(3.5, 1, 0), frame.plot = TRUE, yaxs="r", cex.lab=2.5, las=1, xaxt="n", yaxt="n",fg=grey(0.5), family="serif")
axis(1, at = labels_at, c("1969","1970","1971","1972","1973","1974","1975","1976","1977","1978","1979","1980","1981","1982","1983","1984","1985","1986","1987","1988"), tick = TRUE, lty = 1, mgp= c(3, 1.4, 0), las=1, cex.axis=2.5, font=1, col=grey(0.5), col.ticks=grey(0.3), family="")
axis(2, at = NULL, labels = TRUE, tick = TRUE, lty = 1, mgp= c(3, 0.7, 0), las=1, cex.axis=2.5, font=5, col=grey(0.5), col.ticks=grey(0.3)) #mgp= c(2.5, 0.7, 0)
title(xlab ="Year", mgp= c(3.7, 1, 0), cex.lab=2, las=1)
title(ylab ="Proportion of births over the mean", mgp= c(4.1, 0.7, 0), cex.lab=2, las=1)

# lines(ind, (f[ind,1]*std_y+m_y)/m_y, col="grey", lwd=1)			# f
# lines(c(0,7305), c(1,1), lty=2, lwd=2)
abline(h=1, lty=2)												# mean
lines(ind, (f1[ind,1]*std_y+m_y)/m_y, col=2, lwd=2)				# smooth trend
lines(ind, (f3[ind,1]*std_y+m_y)/m_y, col=3, lwd=2)				# year effect
# lines(ind1[], (f5_BF[[1]][,1]*sd_y+m_y)/m_y, col=6, lwd=1)	# horseshoe

legend("topleft",inset=c(0.25,0.03),legend=c("Observations",TeX('Long-term trend ($f_1$)'),TeX('Year effects ($f_2$)')), col=c(grey(0.4),2,3,"grey"), lty=c(NA,1,1,1), pch=c(20,NA,NA,NA), lwd=c(1,3,3,3), cex=1.7, xpd=TRUE, bty="n", y.intersp=1, x.intersp=0.8, text.font=1, ncol=1, seg.len=1.5)


### PLOT OF ONLY FIRST MOUNTH
dev.new()
par(mai=c(1.02, 1.10, 0.82, 0.42))

data_month <- data[data$month==1&data$year==1972,]
ind <- data_month$id
axis_labels_at <- aggregate(data_month, by=list(data_month$day), FUN=min)$id

# id_week <- data_month[3:9,]$id
id_week <- data_month$id[data_month$day_of_week==1]

plot(ind, (y[ind]*std_y+m_y)/m_y, type="p", pch=21, bg=grey(0.7), cex=1.2, col=grey(0.4), xlab="", ylab="", lwd=1, ylim=c(0.7,1.2), mgp= c(3.5, 1, 0), frame.plot = TRUE, yaxs="r", cex.lab=2.5, las=1, xaxt="n", yaxt="n",fg=grey(0.5), family="serif")
axis(1, at = axis_labels_at, labels = as.character(1:31), tick = TRUE, lty = 1, mgp= c(3, 1.4, 0), las=1, cex.axis=2.5, font=1, col=grey(0.5), col.ticks=grey(0.3), family="")
axis(1, at = id_week, labels = rep(c("Monday"),5), tick = TRUE, lty = 1, mgp= c(-1, -1.2, 0), las=1, cex.axis=1.5, font=1, col=grey(0.5), col.ticks=grey(0.3), family="")
axis(2, at = NULL, labels = TRUE, tick = TRUE, lty = 1, mgp= c(3, 0.7, 0), las=1, cex.axis=2.5, font=5, col=grey(0.5), col.ticks=grey(0.3)) #mgp= c(2.5, 0.7, 0)

title(xlab ="Day", mgp= c(3.7, 1, 0), cex.lab=2, las=1)
title(ylab ="Proportion of births over the mean", mgp= c(4.1, 0.7, 0), cex.lab=2, las=1)

lines(ind, (f[ind,1]*std_y+m_y)/m_y, col="black", lwd=2)	# f
# lines(range(ind), c(1,1), lty=2)	
abline(h=1, lty=2)											# mean
lines(ind, (f1[ind,1]*std_y+m_y)/m_y, col=2, lwd=2)			# f1 smoth trend
lines(ind, (f3[ind,1]*std_y+m_y)/m_y, col=3, lwd=2)	 		# f3 year effect
lines(ind, (f4[ind,1]*std_y+m_y)/m_y, col=4, lwd=2)	 		# f4 week effect
lines(ind, (f5[ind,1]*std_y+m_y)/m_y, col=6, lwd=2) 		# f5 horseshoe

abline(v=id_week, lty=2, col="grey")

#labels special days
text(ind[data_month$month==1&data_month$day==1], y = 0.85, labels = "New year", pos =NULL, offset = 0, family="serif", cex=1.3)

legend("topleft",inset=c(0.12,0.03),legend=c("Observations",TeX('Long-term trend ($f_1$)'),TeX('Year effects ($f_2$)'),TeX('Week effects ($f_3$)'),TeX('Special-days effects ($f_4$)'),TeX('$\\mu$=f_1+f_2+f_3+f_4$')), col=c(grey(0.4),2,3,4,6,"black"), lty=c(NA,1,1,1,1,1), pch=c(20,NA,NA,NA,NA,NA), lwd=c(1,3,3,3,3,3), cex=1.7, xpd=TRUE, bty="n", y.intersp=1, x.intersp=0.8, text.font=1, ncol=2, seg.len=1.3)



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





