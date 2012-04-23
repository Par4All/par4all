// Check pointers to functions

void fpointer01(char         c,
		void       * _stream,
		void      (* my_fputc)(const char c,void * _stream),
		int * col,
		int   indent,
		int * nbout)
{
  if((c == '\n')||(c == '\r'))
    {
      /* on change de ligne */

      *col = 0;
    }
  else
    {
      /* indentation ok ? */

      while(*col < indent)
	{
	  my_fputc(' ',_stream);
	  (*nbout)++;
	  (*col)++;
	}

      (*col)++;
    }

  /* dans tous les cas il faut afficher le caractere passe */

  my_fputc(c,_stream);
  (*nbout)++;
}
