#include <math.h>

#define PI 3.14159265
#define CA 0.0003
#define CB 1.0e-9

float el2(x,qqc,aa,bb)
float x,qqc,aa,bb;
{
	float a,b,c,d,e,f,g,em,eye,p,qc,y,z;
	int l;
	void nrerror();

	if (x == 0.0) return 0.0;
	else if (qqc) {
		qc=qqc;
		a=aa;
		b=bb;
		d=1.0+(c=x*x);
		p=sqrt((1.0+c*qc*qc)/d);
		d=x/d;
		c=d/(p+p);
		z=(eye=a)-b;
		a=0.5*(b+a);
		y=fabs(1.0/x);
		f=0.0;
		l=0;
		em=1.0;
		qc=fabs(qc);
		for (;;) {
			b += (eye*qc);
			g=(e=em*qc)/p;
			d += (f*g);
			f=c;
			eye=a;
			p += g;
			c=0.5*(d/p+c);
			g=em;
			em += qc;
			a=0.5*(b/em+a);
			y -= (e/y);
			if (y == 0.0) y=sqrt(e)*CB;
   			if (fabs(g-qc) <= CA*g) break;
   			qc=sqrt(e)*2.0;
   			l *= 2;
   			if (y<0.0) l++;
   		}
		if (y<0.0) l++;
		e=(atan(em/y)+PI*l)*a/em;
		if (x < 0.0) e = -e;
		return e+c*z;
	} else nrerror("Bad qqc in routine EL2");
}

#undef PI
#undef CA
#undef CB
