/*Source Code From Laure Gonnord*/
//dummy ex from the aspic distribution


int dummy1(){
  
  int t,t0;
  assume(t0>0);
  t=t0;

  while(t<=9) ++t;

  return 0;
}

//Invariant inside the while loop :  {t<t0+10, t0>0, t>=t0}
