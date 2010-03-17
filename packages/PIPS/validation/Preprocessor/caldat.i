#include <math.h>

#define IGREG 2299161

void caldat(julian,mm,id,iyyy)
long julian;
int *mm,*id,*iyyy;
{
	long ja,jalpha,jb,jc,jd,je;

	if (julian >= IGREG) {
		jalpha=((float) (julian-1867216)-0.25)/36524.25;
		ja=julian+1+jalpha-(long) (0.25*jalpha);
	} else
		ja=julian;
	jb=ja+1524;
	jc=6680.0+((float) (jb-2439870)-122.1)/365.25;
	jd=365*jc+(0.25*jc);
	je=(jb-jd)/30.6001;
	*id=jb-jd-(int) (30.6001*je);
	*mm=je-1;
	if (*mm > 12) *mm -= 12;
	*iyyy=jc-4715;
	if (*mm > 2) --(*iyyy);
	if (*iyyy <= 0) --(*iyyy);
}

#undef IGREG
