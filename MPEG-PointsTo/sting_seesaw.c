// Stanford Invariant Generator: seesaw
// http://www.cs.colorado.edu/~srirams/Software/Sting/Examples/LowDim/see-saw.in

// A call to alea should be added since the guards are not exclusive
//
// This impact the result and make it a bit less precise


float alea(void)
{
  return 1.;
}

void sting_seesaw()
{
  int x=0, y=0;

  while(1) {
    // StInG Expected result: x<=2y, y<=3x
    // PIPS results: y<=3x, x<=2y
    // Transition t2
    if(x<=4)
      x++, y+=2;
    // Transition t1
    if(5<=x && x<=7 && alea()>=0.)
      x+=2, y++;
    // Transition t3
    if(7<=x && x<=9 && alea()>=0.)
      x++, y+=3;
    // Transition t4
    if(x>=9)
      x+=2, y++;
    // PIPS result: 4y+5<=13x, y<=3x, x+15<=8y, x<=2y (which implies
    // x>=1 and y >=2)
    // NOP statement get the body postcondition
    x=x;
  }
}
