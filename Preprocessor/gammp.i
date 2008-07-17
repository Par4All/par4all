float gammp(a,x)
float a,x;
{
	float gamser,gammcf,gln;
	void gser(),gcf(),nrerror();

	if (x < 0.0 || a <= 0.0) nrerror("Invalid arguments in routine GAMMP");
	if (x < (a+1.0)) {
		gser(&gamser,a,x,&gln);
		return gamser;
	} else {
		gcf(&gammcf,a,x,&gln);
		return 1.0-gammcf;
	}
}
