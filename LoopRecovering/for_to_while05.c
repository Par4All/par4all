enum { M = 64 };


main(argc,argv)

int argc;
char *argv[];

{

  int l, m, a;

  for(l=0;l<M;l++)
  compute:
    /* Use this do to be able to add a label on a block that is natively
       not possible in PIPS for Fortran compatibility reasons: */
    do {
      for(m=0;m<M;l++)
	a= 0;
    } while (0);
}
