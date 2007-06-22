void mrqcof(x,y,sig,ndata,a,ma,lista,mfit,alpha,beta,chisq,funcs)
float x[],y[],sig[],a[],**alpha,beta[],*chisq;
int ndata,ma,lista[],mfit;
void (*funcs)();	/* ANSI: void (*funcs)(float,float *,float *,float *,int); */
{
	int k,j,i;
	float ymod,wt,sig2i,dy,*dyda,*vector();
	void free_vector();

	dyda=vector(1,ma);
	for (j=1;j<=mfit;j++) {
		for (k=1;k<=j;k++) alpha[j][k]=0.0;
		beta[j]=0.0;
	}
	*chisq=0.0;
	for (i=1;i<=ndata;i++) {
		(*funcs)(x[i],a,&ymod,dyda,ma);
		sig2i=1.0/(sig[i]*sig[i]);
		dy=y[i]-ymod;
		for (j=1;j<=mfit;j++) {
			wt=dyda[lista[j]]*sig2i;
			for (k=1;k<=j;k++)
				alpha[j][k] += wt*dyda[lista[k]];
			beta[j] += dy*wt;
		}
		(*chisq) += dy*dy*sig2i;
	}
	for (j=2;j<=mfit;j++)
		for (k=1;k<=j-1;k++) alpha[k][j]=alpha[j][k];
	free_vector(dyda,1,ma);
}
