struct test
{
  int a;
  int b[2];
};



int main()
{
  struct test t[]=
    {
      {
	1,
	{2, 3},
      },
      {
	1,
	{2, 3},
      }
    };
  return 0;
}
