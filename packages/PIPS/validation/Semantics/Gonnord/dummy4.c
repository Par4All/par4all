/*Source Code From Laure Gonnord*/
//dummy ex from the aspic distribution


int dummy4(){
  
  int t,t0,z;
  assume(t0>0);
  t=t0;
  z=0;

  while(z<=53) {++t;z+=3;};

  return 0;
}

//Invariant inside the while loop : {t0>0, 3t<=3t0+56, t>=t0, 3t=3t0+z}
// at the stop point :  {t>=t0+18, t0>0, 3t<=3t0+56, 3t=3t0+z}

