float erfc(x)
float x;
{
	float gammp(),gammq();

	return x < 0.0 ? 1.0+gammp(0.5,x*x) : gammq(0.5,x*x);
}
