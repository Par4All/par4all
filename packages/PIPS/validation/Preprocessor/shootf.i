void shootf(nvar,v1,v2,delv1,delv2,n1,n2,x1,x2,xf,eps,h1,hmin,f,dv1,dv2)
int nvar,n1,n2;
float v1[],v2[],delv1[],delv2[],x1,x2,xf,eps,h1,hmin,f[],dv1[],dv2[];
{
	int nok,nbad,j,iv,i,*indx,*ivector();
	void odeint(),ludcmp(),lubksb(),derivs(),
		rkqc(),free_matrix(),free_vector(),free_ivector();
	void load1();	/* ANSI: void load1(float,float *,float *); */
	void load2();	/* ANSI: void load2(float,float *,float *); */
	void score();	/* ANSI: void score(float,float *,float *); */
	float sav,det,*y,*f1,*f2,**dfdv,**matrix(),*vector();

	dfdv=matrix(1,nvar,1,nvar);
	y=vector(1,nvar);
	f1=vector(1,nvar);
	f2=vector(1,nvar);
	indx=ivector(1,nvar);
	load1(x1,v1,y);
	odeint(y,nvar,x1,xf,eps,h1,hmin,&nok,&nbad,derivs,rkqc);
	score(xf,y,f1);
	load2(x2,v2,y);
	odeint(y,nvar,x2,xf,eps,h1,hmin,&nok,&nbad,derivs,rkqc);
	score(xf,y,f2);
	j=0;
	for (iv=1;iv<=n2;iv++) {
		j++;
		sav=v1[iv];
		v1[iv] += delv1[iv];
		load1(x1,v1,y);
		odeint(y,nvar,x1,xf,eps,h1,hmin,&nok,&nbad,derivs,rkqc);
		score(xf,y,f);
		for (i=1;i<=nvar;i++)
			dfdv[i][j]=(f[i]-f1[i])/delv1[iv];
		v1[iv]=sav;
	}
	for (iv=1;iv<=n1;iv++) {
		j++;
		sav=v2[iv];
		v2[iv] += delv2[iv];
		load2(x2,v2,y);
		odeint(y,nvar,x2,xf,eps,h1,hmin,&nok,&nbad,derivs,rkqc);
		score(xf,y,f);
		for (i=1;i<=nvar;i++)
			dfdv[i][j]=(f2[i]-f[i])/delv2[iv];
		v2[iv]=sav;
	}
	for (i=1;i<=nvar;i++) {
		f[i]=f1[i]-f2[i];
		f1[i]=(-f[i]);
	}
	ludcmp(dfdv,nvar,indx,&det);
	lubksb(dfdv,nvar,indx,f1);
	j=0;
	for (iv=1;iv<=n2;iv++) {
		v1[iv] += f1[++j];
		dv1[iv]=f1[j];
	}
	for (iv=1;iv<=n1;iv++) {
		v2[iv] += f1[++j];
		dv2[iv]=f1[j];
	}
	free_ivector(indx,1,nvar);
	free_vector(f2,1,nvar);
	free_vector(f1,1,nvar);
	free_vector(y,1,nvar);
	free_matrix(dfdv,1,nvar,1,nvar);
}
