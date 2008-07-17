float erf(x)
float x;
{
	float gammp();

	return x < 0.0 ? -gammp(0.5,x*x) : gammp(0.5,x*x);
}
