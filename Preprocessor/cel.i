#include <math.h>

#define CA 0.0003
#define PIO2 1.57079632679490

float cel(qqc,pp,aa,bb)
float qqc,pp,aa,bb;
{
	float a,b,e,f,g,em,p,q,qc;
	void nrerror();

	if (qqc == 0.0) nrerror("Bad qqc in routine CEL");
	qc=fabs(qqc);
	a=aa;
	b=bb;
	p=pp;
	e=qc;
	em=1.0;
	if (p > 0.0) {
		p=sqrt(p);
		b /= p;
	} else {
		f=qc*qc;
		q=1.0-f;
		g=1.0-p;
		f -= p;
		q *= (b-a*p);
		p=sqrt(f/g);
		a=(a-b)/g;
		b = -q/(g*g*p)+a*p;
	}
	for (;;) {
		f=a;
		a += (b/p);
		g=e/p;
		b += (f*g);
		b += b;
		p=g+p;
		g=em;
		em += qc;
		if (fabs(g-qc) <= g*CA) break;
		qc=sqrt(e);
		qc += qc;
		e=qc*em;
	}
	return PIO2*(b+a*em)/(em*(em+p));
}

#undef CA
#undef PIO2
