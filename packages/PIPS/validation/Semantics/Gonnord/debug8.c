/*Source Code From Laure Gonnord*/
//foo example from the aspic distrib

int random();

int debug8(){
  
  int x,z;
  z=0;
  x=0;

  while(1){
    if (random() && x>=2*z)
      {
	++z;
	++x;
      }
    else {
      z=0;
    }
  }
  return 0;
}

//invariant : {x>=z, x+1>=2z, z>=0}
