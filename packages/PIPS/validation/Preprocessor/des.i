#include <math.h>

typedef struct IMMENSE {unsigned long l,r;} immense;
typedef struct GREAT {unsigned short l,c,r;} great;

unsigned long bit[33];	/* defining declaration */

void des(inp,key,newkey,isw,out)
immense inp,key,*out;
int *newkey,isw;
{
	static char ip[65]=
		{0,58,50,42,34,26,18,10,2,60,52,44,36,
		28,20,12,4,62,54,46,38,30,22,14,6,64,56,48,40,
		32,24,16,8,57,49,41,33,25,17,9,1,59,51,43,35,
		27,19,11,3,61,53,45,37,29,21,13,5,63,55,47,39,
		31,23,15,7};
	static char ipm[65]=
		{0,40,8,48,16,56,24,64,32,39,7,47,15,
		55,23,63,31,38,6,46,14,54,22,62,30,37,5,45,13,
		53,21,61,29,36,4,44,12,52,20,60,28,35,3,43,11,
		51,19,59,27,34,2,42,10,50,18,58,26,33,1,41,9,
		49,17,57,25};
	static great kns[17];
	static int initflag=1;
	int ii,i,j,k;
	unsigned long ic,shifter,getbit();
	immense itmp;
	void cyfun(),ks();

	if (initflag) {
		initflag=0;
		bit[1]=shifter=1L;
		for(j=2;j<=32;j++) bit[j] = (shifter <<= 1);
	}
	if (*newkey) {
		*newkey=0;
		for(i=1;i<=16;i++) ks(key,i,&kns[i]);
	}
	itmp.r=itmp.l=0L;
	for(j=32,k=64;j>=1;j--,k--) {
		itmp.r = (itmp.r <<= 1) | getbit(inp,ip[j],32);
		itmp.l = (itmp.l <<= 1) | getbit(inp,ip[k],32);
	}
	for(i=1;i<=16;i++) {
		ii = (isw == 1 ? 17-i : i);
		cyfun(itmp.l,kns[ii],&ic);
		ic ^= itmp.r;
		itmp.r=itmp.l;
		itmp.l=ic;
	}
	ic=itmp.r;
	itmp.r=itmp.l;
	itmp.l=ic;
	(*out).r=(*out).l=0L;
	for(j=32,k=64;j>=1;j--,k--) {
		(*out).r = ((*out).r <<= 1) | getbit(itmp,ipm[j],32);
		(*out).l = ((*out).l <<= 1) | getbit(itmp,ipm[k],32);
	}
}
