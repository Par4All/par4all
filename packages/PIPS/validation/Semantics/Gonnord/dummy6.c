/*Source Code From Laure Gonnord*/
//dummy ex from the aspic distribution


int dummy6(){
  
  int t,a;
  t=0;
  a=2;

  while(t<=4) {t=a;a=t+1;};

  return 0;
}

//Invariant inside the while loop :{a>=t+1, 2a>=t+4, a<=t+2}
//at the end : {t>=5, a>=t+1, a<=t+2}
