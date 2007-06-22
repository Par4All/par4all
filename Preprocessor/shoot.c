void shoot(nvar,v,delv,n2,x1,x2,eps,h1,hmin,f,dv)
int nvar,n2;
float v[],delv[],x1,x2,eps,h1,hmin,f[],dv[];
{
	int nok,nbad,iv,i,*indx,*ivector();
	float sav,det,*y,**dfdv,**matrix(),*vector();
	void odeint(),ludcmp(),lubksb(),derivs(),rkqc(),
		free_matrix(),free_vector(),free_ivector();
	void load();	/* ANSI: void load(float,float *,float *); */
	void score();	/* ANSI: void score(float,float *,float *); */

	y=vector(1,nvar);
	indx=ivector(1,nvar);
	dfdv=matrix(1,n2,1,n2);
	load(x1,v,y);
	odeint(y,nvar,x1,x2,eps,h1,hmin,&nok,&nbad,derivs,rkqc);
	score(x2,y,f);
	for (iv=1;iv<=n2;iv++) {
		sav=v[iv];
		v[iv] += delv[iv];
		load(x1,v,y);
		odeint(y,nvar,x1,x2,eps,h1,hmin,&nok,&nbad,derivs,rkqc);
		score(x2,y,dv);
		for (i=1;i<=n2;i++)
			dfdv[i][iv]=(dv[i]-f[i])/delv[iv];
		v[iv]=sav;
	}
	for (iv=1;iv<=n2;iv++) dv[iv] = -f[iv];
	ludcmp(dfdv,n2,indx,&det);
	lubksb(dfdv,n2,indx,dv);
	for (iv=1;iv<=n2;iv++) v[iv] += dv[iv];
	free_matrix(dfdv,1,n2,1,n2);
	free_ivector(indx,1,nvar);
	free_vector(y,1,nvar);
}
