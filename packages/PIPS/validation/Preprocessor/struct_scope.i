struct key 
  {
    int node;
  };

main()
{
  int i = 1;
  struct key 
  {
    int node;
  };
  if (i>1)
    {
      struct key
      {
	float node;
      };
    }
  printf("Test\n");
}

int foo()
{
  struct key 
  {
    int node;
  };
  return 1;
}

int toto()
{
  int i = 1;
  if (i>0)
    {
      struct test 
      {
	float a;
      };
    }
}
