void chsone(bins,ebins,nbins,knstrn,df,chsq,prob)
float bins[],ebins[],*df,*chsq,*prob;
int nbins,knstrn;
{
	int j;
	float temp;
	float gammq();
	void nrerror();

	*df=nbins-1-knstrn;
	*chsq=0.0;
	for (j=1;j<=nbins;j++) {
		if (ebins[j] <= 0.0) nrerror("Bad expected number in CHSONE");
		temp=bins[j]-ebins[j];
		*chsq += temp*temp/ebins[j];
	}
	*prob=gammq(0.5*(*df),0.5*(*chsq));
}
