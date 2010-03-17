#include <math.h>

void kstwo(data1,n1,data2,n2,d,prob)
float data1[],data2[],*d,*prob;
int n1,n2;
{
	int j1=1,j2=1;
	float en1,en2,fn1=0.0,fn2=0.0,dt,d1,d2;
	void sort();
	float probks();

	en1=n1;
	en2=n2;
	*d=0.0;
	sort(n1,data1);
	sort(n2,data2);
	while (j1 <= n1 && j2 <= n2) {
		if ((d1=data1[j1]) <= (d2=data2[j2])) {
			fn1=(j1++)/en1;
		}
		if (d2 <= d1) {
			fn2=(j2++)/en2;
		}
		if ((dt=fabs(fn2-fn1)) > *d) *d=dt;
	}
	*prob=probks(sqrt(en1*en2/(en1+en2))*(*d));
}
