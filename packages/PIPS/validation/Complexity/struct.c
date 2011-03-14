#include <stdlib.h>

int main ()
{
  struct Person{
	int Age;
	char Gender;
        };

  struct Person Pierre;

  Pierre.Age = 18;

  Pierre.Gender = 'M';
}
