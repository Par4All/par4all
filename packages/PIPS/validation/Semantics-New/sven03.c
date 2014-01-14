/* To investigate Example 5.4 on Page 13 of INRIA TR 7560
 *
 * I designed this example to kill the simple computation of di and
 * dj.
 *
 * Ideally, I could add it to Vivien's ALICe database and compare
 * results obtained by the different tools...
 *
 * With respect to sven01: n is updated so as not to be used as a
 * parameter by ISL transitive closure.
 *
 * PS: I wonder why initial values are projected out (Francois
 * Irigoin). Property SEMANTICS_FILTER_INITIAL_VALUES has no impact. I
 * add to add variables k0 and n0 to obtain information that used to
 * be available as k#init and n#init...
 */

int main()
{
  int i, j, k, n, k0, n0;

  k = k0, n = n0;
  if(i+j==5) {
    while(k>0) {
      i+=n, j-=n;
      k--;
      n++;
    }
    n++; // For loop exit postcondition
  }
  return 0;
}
