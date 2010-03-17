#define EPS 1.0e-6
#define FREERETURN {free_vector(xj,1,n);free_vector(xi,1,n);\
	free_vector(h,1,n);free_vector(g,1,n);return;}

void sparse(b,n,x,rsq)
float b[],x[],*rsq;
int n;
{
	int j,iter,irst=0;
	float aden,anum,bsq,dgg,eps2,gam,gg,rp;
	float *g,*h,*xi,*xj,*vector();
	void asub(),atsub(),nrerror(),free_vector();

	g=vector(1,n);
	h=vector(1,n);
	xi=vector(1,n);
	xj=vector(1,n);
	eps2=n*EPS*EPS;
	for (;;) {
		++irst;
		asub(x,xi,n);
		rp=bsq=0.0;
		for (j=1;j<=n;j++) {
			bsq += b[j]*b[j];
			xi[j] -= b[j];
			rp += xi[j]*xi[j];
		}
		atsub(xi,g,n);
		for (j=1;j<=n;j++)
			h[j] = g[j] = -g[j];
		for (iter=1;iter<=10*n;iter++) {
			asub(h,xi,n);
			anum=aden=0.0;
			for (j=1;j<=n;j++) {
				anum += g[j]*h[j];
				aden += xi[j]*xi[j];
			}
			if (aden == 0.0) nrerror("Very singular matrix in SPARSE");
			anum /= aden;
			for (j=1;j<=n;j++) {
				xi[j]=x[j];
				x[j] += anum*h[j];
			}
			asub(x,xj,n);
			*rsq=0.0;
			for (j=1;j<=n;j++) {
				xj[j] -= b[j];
				*rsq += xj[j]*xj[j];
			}
			if (*rsq == rp || *rsq <= bsq*eps2) FREERETURN
			if (*rsq > rp) {
				for (j=1;j<=n;j++) x[j]=xi[j];
				if (irst >= 3) FREERETURN
				break;
			}
			rp = *rsq;
			atsub(xj,xi,n);
			gg=dgg=0.0;
			for (j=1;j<=n;j++) {
				gg += g[j]*g[j];
				dgg += (xi[j]+g[j])*xi[j];
			}
			if (gg == 0.0) FREERETURN
			gam=dgg/gg;
			for (j=1;j<=n;j++) {
				g[j] = -xi[j];
				h[j]=g[j]+gam*h[j];
			}
		}
		nrerror("Too many interations in routine SPARSE");
	}
}

#undef EPS
#undef FREERETURN
