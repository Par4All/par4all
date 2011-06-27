/* Source Code From Laure Gonnord
 *
 * Slightly modified to use assert instead of assume.
 *
 * Gulwani, CAV 2008, probably Example 2
 */

#include <assert.h>

int gulwani_cav08_02(int x,int y, int z){
  int y0,k;
  assert(z<0);
  y0 = y;
  k=0;
  while(x>0){
    //    assert(y>=y0-k,"__bad");
    k=k+1;
    x=x+y;
    y=y+z;
  }

  return 0;
}

int gulwani_cav08_02_r(int x,int y, int z){
  int y0,k;
  assert(z<0);
  y0 = y;
  k=0;
  while(x>0){
    while(x>0 && y >= 0) {
    //    assert(y>=y0-k,"__bad");
    k=k+1;
    x=x+y;
    y=y+z;
    }
    while(x>0 && y < 0) {
    //    assert(y>=y0-k,"__bad");
    k=k+1;
    x=x+y;
    y=y+z;
    }
  }

  return 0;
}
