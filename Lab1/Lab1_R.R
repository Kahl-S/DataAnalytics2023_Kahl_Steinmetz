# Import and view EPI2010_data.csv
EPI_data <- read.csv(file = '2010EPI_data.csv', header=TRUE)
View(EPI_data)
# Import and read EPI_2010_data.xls
EPI_data <- read_excel('2010EPI_data.xls',sheet=4)
View(EPI_data)
#Attach and try out 'fix' func
attach(EPI_data)
fix(EPA_data)

#NA value stuff
tf <- is.na(EPI)
E <- EPI[!tf]


#Summary and investigation
summary(EPI_data$EPI)
fivenum(EPI_data$EPI,na.rm=TRUE)
stem(EPI)
hist(EPI)
hist(EPI,seq(30.,95.,1.0),prob=TRUE)
lines(density(EPI,na.rm=TRUE,bw=1.))
rug(EPI)

#ECDF plot
plot(ecdf(EPI),do.points=FALSE,verticals=TRUE)
par(pty='s')

#QQ Norm Plot
qqnorm(EPI);qqline(EPI)

#Other QQ plot
x<-seq(30,95,1)
qqplot(qt(ppoints(250),df=5),x,xlab='Q-Q plot for t dsn')
qqline(x)