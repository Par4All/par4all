/**
	This file contains the tests used to warn users that they may have
	a stacksize problem.
	
	HPC Project 2011 

**/

#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

#define checkStackSize()		checkStackSizeInline (__FILE__, __LINE__)

static inline void checkStackSizeInline(const char *currentFile, const int currentLine)
{
	struct rlimit r;	
	getrlimit(RLIMIT_STACK, &r);	
	if (sizeof(rlim_t) > sizeof(unsigned long long))
        fprintf(stderr, "Warning: file: %s line: %d rlim_t > unsigned long long; results may be wrong\n", currentFile, currentLine);
    unsigned long long stacksize = (unsigned long long)r.rlim_cur;
    unsigned long long stacksize_max = (unsigned long long)r.rlim_max;
    if (stacksize < stacksize_max)
		fprintf(stderr, "Warning: file: %s line: %d your stacksize is %llu you may need to set it to unlimited size using the command 'ulimit -s unlimited' \n", currentFile, currentLine, stacksize);
}
