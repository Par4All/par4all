/* Representation of calling context for argv */

/* This declaration generates a warning in gcc if the function name is
   "main". */

int argv06(int argc, char * (*argv)[argc]) 
{
  char *p = (void *) 0;
  argv++;
  p = (*argv)[2];
  return p==p; // To silence gcc
}
