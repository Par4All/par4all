// Stanford Invariant Generator: berkeley
// http://www.cs.colorado.edu/~srirams/Software/Sting/Examples/LowDim/berkely.in

// A call to alea should be added since the guards are not exclusive
//
// This impact the result and make it a bit less precise

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

// I assume it returns a value between 0. and 1., but I haven't
// checked that rand(0 was always positive
float alea(void)
{
  float fr = ((float) rand())/((float)RAND_MAX);
  //printf("fr=%f\n", fr);
  return fr;
  //return 1.;
}

void sting_berkeley()
{
  //
  // BERKELEY model taken from Fast
  //

  int invalid, unowned=0, nonexclusive=0, exclusive=0;
  int o_invalid = 0, o_unowned=unowned, o_nonexclusive=nonexclusive,
    o_exclusive=exclusive;
  int count = 0;
  int init_invalid;
#define COUNT_MAX 1000

  //propsteps(3)

  scanf("%d", &invalid);
  init_invalid = invalid;

  assert(invalid>=1);

  srand(1);

  //  while(1) {
  while(count<=COUNT_MAX &&(invalid>=1 || nonexclusive+unowned>=1)) {
    // StInG expected results: exclusive>=0, unowned>=0,
    // invalid+unowned+nonexclusive+exclusive>=1

    // Experimental invariants:
    // exclusive+nonexclusive<=1
    // invalid+unowned+nonexclusive+exclusive==init_invalid
    // exclusive>=0, nonexclusive>=0, invalid>=0, unowned>=0

    // Add type information
    assert(exclusive>=0);
    assert(nonexclusive>=0);
    assert(invalid>=0);
    assert(unowned>=0);


    if(invalid==o_invalid  && unowned==o_unowned
       && nonexclusive==o_nonexclusive && exclusive==o_exclusive) {
      //printf("useless\n");
      ;
    }
    else {
      printf("exclusive=%d, unowned=%d, nonexclusive=%d, invalid=%d\n",
	     exclusive, unowned, nonexclusive, invalid);
      o_invalid = invalid, o_unowned=unowned, o_nonexclusive=nonexclusive,
	o_exclusive=exclusive;
      count++;
      // check the invariants
      assert(exclusive+nonexclusive<=1);
      assert(invalid+unowned+nonexclusive+exclusive==init_invalid);
      assert(exclusive>=0);
      assert(nonexclusive>=0);
      assert(invalid>=0);
      assert(unowned>=0);
    }

    //transition t1: l0,
    if(invalid >= 1 && alea()>=0.5)
      nonexclusive+=exclusive, exclusive=0, invalid--, unowned++;

    //transition t2: l0,
    if(nonexclusive+unowned>=1 && alea()>=0.5)
      invalid+=unowned+nonexclusive-1, exclusive++, unowned=0, nonexclusive=0;

    //transition t3: l0,
    if(invalid >= 1 && alea()>=0.5)
      invalid+=unowned+exclusive+nonexclusive-1, unowned=0, nonexclusive=0,
	exclusive=1;
    // PIPS result: 4y+5<=13x, y<=3x, x+15<=8y, x<=2y (which implies
    // x>=1 and y >=2)
    // NOP statement get the body postcondition
    unowned=unowned;
    //count++;
  }
  // This point should never be reached, but it is. Try with small
  // initial values for invalid
  printf("count = %d\n", count);
}

main(){
  sting_berkeley();
}
