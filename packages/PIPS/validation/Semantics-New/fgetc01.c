#include <stdlib.h>
#include <stdio.h>

#define N 64

float imagein_re[N][N];
float imagein_im[N][N];


void getimage(void)
{
  int i,j;
  unsigned char c;
  FILE *fp;

  /* To try to avoid fp privatization */
  fp = NULL;

  /* This loops should not be distributed.  On 2009/05/26, if the main()
     is removed, this bug too...

     FI: the function is never called and thus everything is parallel
     and loops can e distributed.
  */
  for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
      c=fgetc(fp);
      imagein_re[i][j]=(c==0) ? 0.0 : 1.0;
      imagein_im[i][j]=0.0;
    }
}

int main(argc,argv)

int argc;
char *argv[];

{
  //getimage();
  return EXIT_SUCCESS;
}
