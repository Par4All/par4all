/* Functions used by the PIPS code generation functions. We also need
   an idiv function */

int pips_min(int count, int first,  ...)
{
    va_list ap;
    int j;
    int min = first;
    va_start(ap, first);
    for(j=0; j<count-1; j++) {
        int next =va_arg(ap, int);
	min = min<next? min : next;
    }
    va_end(ap);
    return min;
}

int pips_max(int count, int first,  ...)
{
    va_list ap;
    int j;
    int max = first;
    va_start(ap, first);
    for(j=0; j<count; j++) {
        int next =va_arg(ap, int);
	max = max>next? max : next;
    }
    va_end(ap);
    return max;
}
