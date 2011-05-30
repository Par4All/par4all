/* test case for use-def elimination suggested by François for Mehdi.*/
int dead_code_elim01()
{
  int i, *x, *y;

// i is firstly initialized
  i = 2;

//  make y points to i
  x = &i;
  y = x;

// Here we kill first i assignment
  *y = 1; 
  
  return i;
}

