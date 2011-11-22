// Stanford Invariant Generator: berkeley
// http://www.cs.colorado.edu/~srirams/Software/Sting/Examples/LowDim/berkely.in

// A call to alea should be added since the guards are not exclusive

// The tests in the loop are replaced by while loops

#include <assert.h>

float alea(void)
{
  return 1.;
}

void sting_berkeley02()
{
  //
  // BERKELEY model taken from Fast
  //

  int invalid, unowned=0, nonexclusive=0, exclusive=0;

  //propsteps(3)


  assert(invalid>=1);

  while(1) {
    // StInG expected results: exclusive>=0, unowned>=0,
    // invalid+unowned+nonexclusive+exclusive>=1


    //transition t1: l0,
    while(invalid >= 1 && alea()>=0.)
      nonexclusive+=exclusive, exclusive=0, invalid--, unowned++;

    //transition t2: l0,
    while(nonexclusive+unowned>=1 && alea()>=0.)
      invalid+=unowned+nonexclusive-1, exclusive++, unowned=0, nonexclusive=0;

    //transition t3: l0,
    while(invalid >= 1 && alea()>=0.)
      invalid+=unowned+exclusive+nonexclusive-1, unowned=0, nonexclusive=0, exclusive=1;
    // PIPS result: 4y+5<=13x, y<=3x, x+15<=8y, x<=2y (which implies
    // x>=1 and y >=2)
    // NOP statement get the body postcondition
    unowned=unowned;
  }
}
