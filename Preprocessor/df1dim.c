extern int ncom;	/* defined in DLINMIN */
extern float *pcom,*xicom,(*nrfunc)();
extern void (*nrdfun)();

float df1dim(x)
float x;
{
	int j;
	float df1=0.0;
	float *xt,*df,*vector();
	void free_vector();

	xt=vector(1,ncom);
	df=vector(1,ncom);
	for (j=1;j<=ncom;j++) xt[j]=pcom[j]+x*xicom[j];
	(*nrdfun)(xt,df);
	for (j=1;j<=ncom;j++) df1 += df[j]*xicom[j];
	free_vector(df,1,ncom);
	free_vector(xt,1,ncom);
	return df1;
}
