#define IM 11979
#define IA 430
#define IC 2531
#define NACC 24
#define IB1 1L
#define IB3 4L
#define IB4 8L
#define IB32 0x80000000L
#define MASK IB1+IB3+IB4

typedef struct IMMENSE {unsigned long l,r;} immense;

float ran4(idum)
int *idum;
{
	static int newkey,iff=0;
	static immense inp,key,jot;
	static double pow[66];
	unsigned long isav,isav2;
	int j;
	double r4;
	void des();

	if (*idum < 0 || iff == 0) {
		iff=1;
		*idum %= IM;
		if (*idum < 0) *idum += IM;
		pow[1]=0.5;
		key.r=key.l=inp.r=inp.l=0L;
		for (j=1;j<=64;j++) {
			*idum = ((long) (*idum)*IA+IC) % IM;
			isav=2*(unsigned long)(*idum)/IM;
			if (isav) isav=IB32;
			isav2=(4*(unsigned long)(*idum)/IM) % 2;
			if (isav2) isav2=IB32;
			if (j <= 32) {
				key.r=(key.r >>= 1) | isav;
				inp.r=(inp.r >>= 1) | isav2;
			} else {
				key.l=(key.l >>= 1) | isav;
				inp.l=(inp.l >>= 1) | isav2;
			}
			pow[j+1]=0.5*pow[j];
		}
		newkey=1;
	}
	isav=inp.r & IB32;
	if (isav) isav=1L;
	if (inp.l & IB32)
		inp.r=((inp.r ^ MASK) << 1) | IB1;
	else
		inp.r <<= 1;
	inp.l=(inp.l << 1) | isav;
	des(inp,key,&newkey,0,&jot);
	r4=0.0;
	for (j=1;j<=NACC;j++) {
		if (jot.r & IB1) r4 += pow[j];
		jot.r >>= 1;
	}
	return r4;
}

#undef IM
#undef IA
#undef IC
#undef NACC
#undef IB1
#undef IB3
#undef IB4
#undef IB32
#undef MASK
