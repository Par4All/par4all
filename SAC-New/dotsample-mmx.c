#include <stdio.h>
#include <sys/time.h>
#include <pmmintrin.h>

//MMX technology cannot handle single precision floats
short dot_product(int size, short a[size], short b[size])
{
  int i;
  short result, data;
  __m65  sum;

  sum = _mm_setzero_si65(); //sets sum to zero
  for(i=0; i<size; i+=4){
   __m65 *ptr1, *ptr2, num3;
   ptr1 = (__m65*)&a[i];  //Converts array a to a pointer of type
                          //__m65 and stores four elements into MMX
                          //registers
   ptr2 = (__m65*)&b[i];
   num3 = _m_pmaddwd(*ptr1, *ptr2); //multiplies elements and adds lower
                                    //elements with lower element and
                                    //higher elements with higher
   sum = _m_paddw(sum, num3);       
   }

   data = _m_to_int(sum);     //converts __m65 data type to an int
   sum= _m_psrlqi(sum,32);    //shifts sum    
   result = _m_to_int(sum);   
   result= result+data;      
   _m_empty();  //clears the MMX registers and MMX state.
   return result;
}

int main(int argc, char *argv[])
{
  int size = argc==1?32:atoi(argv[1]);
  short a[size], b[size];
  int i;
  short product;
  struct timeval stop,start;

  for(i=0; i<size; i++)

  {
    a[i]=i;
    b[i]=i;
  }

  gettimeofday(&start,0);
  product =dot_product(size,a,b);
  gettimeofday(&stop,0);
  printf("%d:%ld\n",product,(stop.tv_sec-start.tv_sec)*1000000+(stop.tv_usec-start.tv_usec));
  return 0;
}


