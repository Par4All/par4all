/*Source Code From Laure Gonnord*/
//example from Cousot/Halbwachs's paper 1971

int random();

int hal79a(){
  
  int i,j;
  i=0;
  j=0;

  while(i<=100){
    if (random()) i=i+4;
    else {i=i+2;j=j+1;}
  }

  return 0;
}

//Invariant inside the while loop : {i>=2j, j>=0, i<=104, i+2j<=204}  
//at the end : {i>=101, i>=2j, j>=0, i<=104, i+2j<=204} 
//aspic -delay 54 (accel of circuits not yet implemented)
