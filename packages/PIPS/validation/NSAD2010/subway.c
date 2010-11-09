//  Subway example 
// The train detects beacons that are placed along the trasck, 
// and receives the "second" from a central clock.
// The train adjusts its speed as:
//  - B-S >= 10 it is early and puts brake as long as B>S
//              it  must stop before encountering 10 beacons
//  - B-S <= 10 it is late and is
//              considered late as long B <S. 


#include <stdio.h>
#include <stdlib.h>

float alea(void)
{
  float fr = ((float) rand())/((float)RAND_MAX);
  return (fr);
}

int main()
{
  int s=0; // the number of seconds
  int b=0; // the number of beacons 
  int d=0; // the  number of beacons after
	   // the train begins to brake
  
  //  while(1) {
  
  // The train is ON TIME
 ontime:
  if (alea()>0.) b++;
  if (alea()>0.) s++;
  if (s-b>=10) goto late;
  if(s-b<=-10) goto early;
  goto ontime;
  
  // the train is LATE
 late:
  if (alea()>0.) b++;
  if (s==b) goto ontime;
  goto late;
  
  // the train is early
 early:
  d=0;
 
 brake:
  // the train puts brake while(s!=b) and d<=10
  if (alea()>0.) {b++;d++;}
  if (alea()>0.) s++;
  if (s==b) goto ontime;
  if (d>=10) goto stopped;
  goto brake;
  
  // It stops and waits after 10 beacons
 stopped:
  if (alea()>0.) s++;
  if (s==b) goto ontime;
  goto stopped;
  
  // }
}

