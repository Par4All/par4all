/* Warning: getimage wss never called. Anything can happen!
 *
 * FI: I remove "main" to check the privatization of scalar c
 */

#include <stdlib.h>
#include <stdio.h>

#define N 64

float imagein_re[N][N];
float imagein_im[N][N];

unsigned char foo(FILE * fp)
{
  unsigned char c = 0;
  // Bug with proper effects:
  //return c = * ((unsigned char *) fp);
  fp++; // The write effect is lost, but not the read effect, as long
	// use-def elimination is not used
  return c;
}

void getimage(void)
{
  int i,j;
  unsigned char c;
  FILE *fp;

  /* To try to avoid fp privatization */
  fp = NULL;

  /* This loops should not be distributed.  On 2009/05/26, if the main()
     is removed, this bug too... */
  for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
      c=foo(fp);
      imagein_re[i][j]=(c==0) ? 0.0 : 1.0;
      imagein_im[i][j]=0.0;
    }
}
/*
int main(argc,argv)

int argc;
char *argv[];

{
	return EXIT_SUCCESS;
}
*/
