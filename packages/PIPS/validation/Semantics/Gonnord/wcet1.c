/*Source Code From Laure Gonnord*/
//example  wcet 1 from aspic distrib

int random();

int wcet1(){
  
  int i,j,k,a;

  assume(0 <= i && 0 <=a && a<=5 && j==0 && k==0);
  
  while(i<=2 && j<=9){
    if (random()) j=j+1;
    else {a=a+2;j=j+1;k=k+1;}
  }

  return 0;
}

//Invariant inside the while loop : 
