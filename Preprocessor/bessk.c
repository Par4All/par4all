float bessk(n,x)
int n;
float x;
{
	int j;
	float bk,bkm,bkp,tox;
	float bessk0(),bessk1();
	void nrerror();

	if (n < 2) nrerror("Index n less than 2 in BESSK");
	tox=2.0/x;
	bkm=bessk0(x);
	bk=bessk1(x);
	for (j=1;j<n;j++) {
		bkp=bkm+j*tox*bk;
		bkm=bk;
		bk=bkp;
	}
	return bk;
}
