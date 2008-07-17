#include <math.h>

#define IGREG (15+31L*(10+12L*1582))

long julday(mm,id,iyyy)
int mm,id,iyyy;
{
	long jul;
	int ja,jy,jm;
	void nrerror();

	if (iyyy == 0) nrerror("JULDAY: there is no year zero.");
	if (iyyy < 0) ++iyyy;
	if (mm > 2) {
		jy=iyyy;
		jm=mm+1;
	} else {
		jy=iyyy-1;
		jm=mm+13;
	}
	jul = (long) (floor(365.25*jy)+floor(30.6001*jm)+id+1720995);
	if (id+31L*(mm+12L*iyyy) >= IGREG) {
		ja=0.01*jy;
		jul += 2-ja+(int) (0.25*ja);
	}
	return jul;
}

#undef IGREG
