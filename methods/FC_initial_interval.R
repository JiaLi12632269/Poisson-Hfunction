b<-3
a5<-vector()
a6<-vector()
mu0<-seq(0,1000,0.005)
for(mu in mu0){
      a1<-vector()
      a2<-vector()
      ymax<-qpois(0.999999,mu+b)
      y<-seq(0,ymax,1)
      for(n in y){
           mubst<-max(0,n-b)
           p1<-dpois(n,mu+b)
           p2<-dpois(n,mubst+b)
           R<-p1/p2
           a1<-c(a1,R)
           a2<-c(a2,p1)
      }
      ran<-order(a1,decreasing=TRUE)
      ranp<-a2[ran]
      rann<-y[ran]
      pp<-0
      a3<-vector()
      for(i in 1:length(y)){
            pp<-pp+ranp[i]
            a3<-c(a3,pp)
      }
      cc<-min(which(a3>=0.9))
      yy<-rann[1:cc]
      nL<-min(yy)
      nU<-max(yy)
      a5<-c(a5,nL)
      a6<-c(a6,nU)
}
a7<-vector()
a8<-vector()
n1<-0:600
for(n in n1){
      index<-which(a5<=n&n<=a6)
      muL<-min(mu0[index])
      muU<-max(mu0[index])
      a7<-c(a7,muL)
      a8<-c(a8,muU)
}
matrix<-rbind(n1,a7, a8)
round(matrix,2)
