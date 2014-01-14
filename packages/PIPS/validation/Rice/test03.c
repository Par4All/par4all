/* Try casts
 *
 * We have a problem because we want all points-to information to be
 * proprely typed. The bucket allocated for "toto" is identified as an
 * arry of double elements. But "toto" is a "void *" and won't carry
 * the previous information explicitly. Furthermore, the bucket is
 * casted as a 2-D array of double elements.
 *
 * We could refuse the cast and return an anywhere cell for toto. Or
 * we should check the compatibility between the cast type and the
 * type of the cell pointed to by "toto".
 */

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv){
  int a_d1, a_d2;
  int i, j;
  void * toto;

  a_d1=a_d2=5;
  // a_d1*a_d2 must be parenthesized because of the limited number of
  // patterns checked by the abstract heap stub generator
  toto = malloc(sizeof(double)*(a_d1*a_d2));

  for(i=0;i<a_d1;i++){
    for(j=0;j<a_d2;j++){
      (*(double(*)[a_d1][a_d2])toto)[i][j] = (double) i+j*a_d2;
    }
  }

  free(toto);

  a_d1=a_d2=10;
  toto = malloc(sizeof(double)*(a_d1*a_d2));

  for(i=0;i<a_d1;i++){
    for(j=0;j<a_d2;j++){
      (*(double(*)[a_d1][a_d2])toto)[i][j] = (double) i+j*a_d2;
    }
  }

  free(toto);

  return 0;
}
