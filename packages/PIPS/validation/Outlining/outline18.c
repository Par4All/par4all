
#define size 100

typedef struct {
  int val;
} my_type;


///@brief update particles features
int main (void) {
  my_type t[10];

  for(int i = 0; i < 10; i ++) {
	t[i].val= i;
  }

  // go through my types
  for(int i = 0; i < 10; i ++) {
	int v = t[i].val;
  kernel:	  v = i;
  }
  return 0;
}
