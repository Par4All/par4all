enum { M = 64 };

main(argc,argv)

int argc;
char *argv[];

{

  int l, m, a;

  for(l=0; l<M; l++)
    /* Since this for loop is desugared in a while() with an
       initialization before, it is a sequence, thus the label which
       cannor be associated on a sequence is put on a ";" inside the
       sequence and this is the ";" which is outlined instead of the
       for-loop... */
  compute:
    for(m=0; m<M; m = 2*m)
      a= 0;
}
