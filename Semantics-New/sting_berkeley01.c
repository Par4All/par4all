// Stanford Invariant Generator: berkeley
// http://www.cs.colorado.edu/~srirams/Software/Sting/Examples/LowDim/berkely.in

// A call to alea should be added since the guards are not exclusive
//
// This impact the result and make it a bit less precise

// Problem: the while loop body may have no effect on the states.

#include <assert.h>

float alea(void)
{
  return 1.;
}

void sting_berkeley01()
{
  //
  // BERKELEY model taken from Fast
  //

  int invalid, unowned=0, nonexclusive=0, exclusive=0;

  //propsteps(3)


  assert(invalid>=1);

  while(invalid>=1 || nonexclusive+unowned>=1) {
    // StInG expected results: exclusive>=0, unowned>=0,
    // invalid+unowned+nonexclusive+exclusive>=1

    //transition t1 or t2 or t3 may execute: l0,
    if(invalid >= 1 && nonexclusive+unowned>=1) {
      if(alea()>=0.)
	// t1
	nonexclusive+=exclusive, exclusive=0, invalid--, unowned++;
      else if(alea()>=0)
	// t2
	invalid+=unowned+nonexclusive-1, exclusive++, unowned=0, nonexclusive=0;
      else
	// t3
	invalid+=unowned+exclusive+nonexclusive-1, unowned=0, nonexclusive=0, exclusive=1;
    }
    //transition t1 or t3 may execute but not t2: l0,
    else if(invalid >= 1) {
      assert(!(nonexclusive+unowned>=1));
      if(alea()>=0.)
	//transition t1: l0,
	nonexclusive+=exclusive, exclusive=0, invalid--, unowned++;
      else
	//transition t3: l0,
	invalid+=unowned+exclusive+nonexclusive-1, unowned=0, nonexclusive=0, exclusive=1;
    }
    else //if(nonexclusive+unowned>=1)
      assert(!(invalid>=1) && nonexclusive+unowned>=1);
      //transition t2: l0 must be executable,
	invalid+=unowned+nonexclusive-1, exclusive++, unowned=0, nonexclusive=0;
    //else
      // This point should never be reached
      //abort(0);
  }
  // This point should never be reached, but it is. Try with small
  // initial values for invalid
  printf("count = %d\n", count);
}
