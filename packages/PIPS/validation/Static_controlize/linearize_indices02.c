

void not_linearized_array_indices(int n, float fx[n], float pot[n])
{
   for(int i = 0; i <= n; i += 1)
     fx[i] = pot[((i+1)&128)-1]/(2.*(float) 6.f/128);
}

