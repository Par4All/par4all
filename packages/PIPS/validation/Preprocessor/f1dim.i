extern int ncom;	/* defined in LINMIN */
extern float *pcom,*xicom,(*nrfunc)();

float f1dim(x)
float x;
{
	int j;
	float f,*xt,*vector();
	void free_vector();

	xt=vector(1,ncom);
	for (j=1;j<=ncom;j++) xt[j]=pcom[j]+x*xicom[j];
	f=(*nrfunc)(xt);
	free_vector(xt,1,ncom);
	return f;
}
