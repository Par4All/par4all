/* $Id$ */
/* a la java StringBuffer */

#include <stdlib.h>
#include "genC.h"

typedef struct __string_buffer_head
{
	stack ins;
}
  _string_buffer_head;

string_buffer string_buffer_make(void)
{
	string_buffer n = (string_buffer) malloc(sizeof(_string_buffer_head));
	message_assert("n allocated", n!=NULL);
	n->ins = stack_make(0, 0, 0);
	return n;
}

void string_buffer_free(string_buffer *psb, bool free_strings)
{
	if (free_strings)
		STACK_MAP_X(s, string, free(s), (*psb)->ins, 0);
	stack_free(&((*psb)->ins));
	free(*psb);
	*psb = NULL;
}

string string_buffer_to_string(string_buffer sb)
{
	int bufsize = 0, current = 0;
	char * buf = NULL;

	STACK_MAP_X(s, string, bufsize+=strlen(s), sb->ins, 0);

	buf = (char*) malloc(sizeof(char)*(bufsize+1));
	buf[current] = '\0';

	STACK_MAP_X(s, string, 
	{
		int len = strlen(s);
		(void) memcpy(&buf[current], s, len);
		current += len;
		buf[current] = '\0';
	},
				sb->ins, 0);
	
	return buf;
}

void string_buffer_append(string_buffer sb, string s)
{
	stack_push(sb->ins, (void*) s);
}
