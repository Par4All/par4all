#define TOL 2.0e-4

int ncom=0;	/* defining declarations */
float *pcom=0,*xicom=0,(*nrfunc)();
void (*nrdfun)();

void dlinmin(p,xi,n,fret,func,dfunc)
float p[],xi[],*fret,(*func)();
void (*dfunc)();
int n;
{
	int j;
	float xx,xmin,fx,fb,fa,bx,ax;
	float dbrent(),f1dim(),df1dim(),*vector();
	void mnbrak(),free_vector();

	ncom=n;
	pcom=vector(1,n);
	xicom=vector(1,n);
	nrfunc=func;
	nrdfun=dfunc;
	for (j=1;j<=n;j++) {
		pcom[j]=p[j];
		xicom[j]=xi[j];
	}
	ax=0.0;
	xx=1.0;
	bx=2.0;
	mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
	*fret=dbrent(ax,xx,bx,f1dim,df1dim,TOL,&xmin);
	for (j=1;j<=n;j++) {
		xi[j] *= xmin;
		p[j] += xi[j];
	}
	free_vector(xicom,1,n);
	free_vector(pcom,1,n);
}

#undef TOL
