void zbrak(fx,x1,x2,n,xb1,xb2,nb)
float x1,x2,xb1[],xb2[];
float (*fx)();	/* ANSI: float (*fx)(float); */
int n,*nb;
{
	int nbb,i;
	float x,fp,fc,dx;

	nbb=(*nb);
	*nb=0;
	dx=(x2-x1)/n;
	fp=(*fx)(x=x1);
	for (i=1;i<=n;i++) {
		fc=(*fx)(x += dx);
		if (fc*fp < 0.0) {
			xb1[++(*nb)]=x-dx;
			xb2[*nb]=x;
		}
		fp=fc;
		if (nbb == (*nb)) return;
	}
}
