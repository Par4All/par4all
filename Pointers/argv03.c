/* Representation of calling context for argv */

/* This declaration generates a warning in gcc if the function name is
   "main". */

int argv03(int argc, char * (*argv)[argc]) 
{
  char *p = (*argv)[1];
  return p==p; // To silence gcc
}
