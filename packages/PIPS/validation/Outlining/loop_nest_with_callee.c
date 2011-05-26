
void my_func(int toto, int  titi[toto])
{
 // Do nothing
}



void another_func( )
{
  int tv = 10;
  int k, n=10;
  int some_array[n][n];


kernel:    for(k=0;k<n;k++) {
      my_func(n,some_array[n]);
    }

}


int main()
{
  another_func();
  return(1);
}


