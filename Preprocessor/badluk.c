#include <stdio.h>
#include <math.h>

#define ZON -5.0
#define IYBEG 1900
#define IYEND 2000

main()  /* Program BADLUK */
{
	int ic,icon,idwk,im,iyyy,n;
	float timzon = ZON/24.0,frac;
	long jd,jday;
	void flmoon();
	long julday();

	printf("\nFull moons on Friday the 13th from %5d to %5d\n",IYBEG,IYEND);
	for (iyyy=IYBEG;iyyy<=IYEND;iyyy++) {
		for (im=1;im<=12;im++) {
			jday=julday(im,13,iyyy);
			idwk=(int) ((jday+1) % 7);
			if (idwk == 5) {
				n=12.37*(iyyy-1900+(im-0.5)/12.0);
				icon=0;
				for (;;) {
					flmoon(n,2,&jd,&frac);
					frac=24.0*(frac+timzon);
					if (frac < 0.0) {
						--jd;
						frac += 24.0;
					}
					if (frac > 12.0) {
						++jd;
						frac -= 12.0;
					} else
						frac += 12.0;
					if (jd == jday) {
						printf("\n%2d/13/%4d\n",im,iyyy);
						printf("%s %5.1f %s\n","Full moon",frac,
							" hrs after midnight (EST)");
						break;
					} else {
						ic=(jday >= jd ? 1 : -1);
						if (ic == (-icon)) break;
						icon=ic;
						n += ic;
					}
				}
			}
		}
	}
}
