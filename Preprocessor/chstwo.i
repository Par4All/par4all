void chstwo(bins1,bins2,nbins,knstrn,df,chsq,prob)
float bins1[],bins2[],*df,*chsq,*prob;
int nbins,knstrn;
{
	int j;
	float temp;
	float gammq();

	*df=nbins-1-knstrn;
	*chsq=0.0;
	for (j=1;j<=nbins;j++)
		if (bins1[j] == 0.0 && bins2[j] == 0.0)
			*df -= 1.0;
		else {
			temp=bins1[j]-bins2[j];
			*chsq += temp*temp/(bins1[j]+bins2[j]);
		}
	*prob=gammq(0.5*(*df),0.5*(*chsq));
}
