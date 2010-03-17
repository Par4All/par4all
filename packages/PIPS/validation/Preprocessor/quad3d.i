static float xsav,ysav;
static float (*nrfunc)(); /* ANSI: static float (*nrfunc)(float,float,float); */

float quad3d(func,x1,x2)
float x1,x2,(*func)();
{
	float qgaus(),f1();

	nrfunc=func;
	return qgaus(f1,x1,x2);
}

float f1(x)
float x;
{
	float qgaus(),f2();
	float yy1(),yy2();	/* ANSI: float yy1(float),yy2(float); */

	xsav=x;
	return qgaus(f2,yy1(x),yy2(x));
}

float f2(y)
float y;
{
	float qgaus(),f3();
	float z1(),z2(); /* ANSI: float z1(float,float),z2(float,float); */

	ysav=y;
	return qgaus(f3,z1(xsav,y),z2(xsav,y));
}

float f3(z)
float z;
{
	return (*nrfunc)(xsav,ysav,z);
}
