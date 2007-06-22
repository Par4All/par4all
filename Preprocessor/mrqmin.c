void mrqmin(x,y,sig,ndata,a,ma,lista,mfit,covar,alpha,chisq,funcs,alamda)
float x[],y[],sig[],a[],**covar,**alpha,*chisq,*alamda;
int ndata,ma,lista[],mfit;
void (*funcs)();
{
	int k,kk,j,ihit;
	static float *da,*atry,**oneda,*beta,ochisq;
	float *vector(),**matrix();
	void mrqcof(),gaussj(),covsrt(),nrerror(),free_matrix(),free_vector();

	if (*alamda < 0.0) {
		oneda=matrix(1,mfit,1,1);
		atry=vector(1,ma);
		da=vector(1,ma);
		beta=vector(1,ma);
		kk=mfit+1;
		for (j=1;j<=ma;j++) {
			ihit=0;
			for (k=1;k<=mfit;k++)
				if (lista[k] == j) ihit++;
			if (ihit == 0)
				lista[kk++]=j;
			else if (ihit > 1) nrerror("Bad LISTA permutation in MRQMIN-1");
		}
		if (kk != ma+1) nrerror("Bad LISTA permutation in MRQMIN-2");
		*alamda=0.001;
		mrqcof(x,y,sig,ndata,a,ma,lista,mfit,alpha,beta,chisq,funcs);
		ochisq=(*chisq);
	}
	for (j=1;j<=mfit;j++) {
		for (k=1;k<=mfit;k++) covar[j][k]=alpha[j][k];
		covar[j][j]=alpha[j][j]*(1.0+(*alamda));
		oneda[j][1]=beta[j];
	}
	gaussj(covar,mfit,oneda,1);
	for (j=1;j<=mfit;j++)
		da[j]=oneda[j][1];
	if (*alamda == 0.0) {
		covsrt(covar,ma,lista,mfit);
		free_vector(beta,1,ma);
		free_vector(da,1,ma);
		free_vector(atry,1,ma);
		free_matrix(oneda,1,mfit,1,1);
		return;
	}
	for (j=1;j<=ma;j++) atry[j]=a[j];
	for (j=1;j<=mfit;j++)
		atry[lista[j]] = a[lista[j]]+da[j];
	mrqcof(x,y,sig,ndata,atry,ma,lista,mfit,covar,da,chisq,funcs);
	if (*chisq < ochisq) {
		*alamda *= 0.1;
		ochisq=(*chisq);
		for (j=1;j<=mfit;j++) {
			for (k=1;k<=mfit;k++) alpha[j][k]=covar[j][k];
			beta[j]=da[j];
			a[lista[j]]=atry[lista[j]];
		}
	} else {
		*alamda *= 10.0;
		*chisq=ochisq;
	}
	return;
}
