#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <polylib/polylib.h>


void Union_Read( Polyhedron **P, Polyhedron **C, char ***param_name )
{
	Matrix *pm;
	Polyhedron *ptmp;
	unsigned NbRows, NbColumns;
	char s[1025], param[1025];
	int i, j, c, f;

	*P = NULL;
	pm = Matrix_Read();
	f=1;
	while( f )
	{
		do
		{
			if( fgets(s, 1024, stdin) == 0  )
				f=0;
		}
		while ( (*s=='#' || *s=='\n') && f );

		if( f && sscanf(s, "%d %d", &NbRows, &NbColumns)==2 )
		{
			/* recupere l'ancien pm et l'ajoute a la liste de matrices */
			if( *P )
				if( pm->NbColumns != ((*P)->Dimension)+2 )
				{
					fprintf( stderr,
						"Polyhedra must be in the same dimension space !\n");
					exit(0);
				}
			ptmp = Constraints2Polyhedron(pm, 200);
			ptmp->next = *P;
			*P = ptmp;
			Matrix_Free(pm);

			/* lit la nouvelle matrice en entree */
			pm = Matrix_Alloc(NbRows, NbColumns);
			Matrix_Read_Input( pm );
		}
		else
			break;
	}

	*C = Constraints2Polyhedron(pm, 200);
	Matrix_Free(pm);


	if( f )
	{
		/* lecture des noms des parametres */
		*param_name = (char **)malloc( (*C)->Dimension*sizeof(char *) );
		c = 0;
		for( i=0 ; i<(*C)->Dimension ; ++i )
		{
			j=0;
			for( ; ; ++c )
			{
				if( s[c]==' ' || s[c]=='\n' || s[c]==0 ) {
					if( j==0 )
						continue;
					else
						break;
				}
				param[j++] = s[c];
			}

			/* si pas assez de parametres, fin (parametres par defaut) */
			if( j==0 )
				break;
			param[j] = 0;
			(*param_name)[i] = (char *)malloc( j );
			strcpy( (*param_name)[i], param );
		}
		if( i != (*C)->Dimension )
		{
			free( *param_name );
			*param_name = NULL;
		}
	}
	else
		*param_name = NULL;

}



int main( int argc, char **argv)
{
	Polyhedron *P, *C;
	char **param_name;
	Enumeration *e, *en;

	if( argc != 1 )
	{
		fprintf( stderr, " Usage : %s [< file]\n", argv[0] );
		return( -1 );
	}

	Union_Read( &P, &C, &param_name );

	e = Domain_Enumerate( P, C, 200, param_name );

	for( en=e ; en ; en=en->next )
	{
	  Print_Domain(stdout,en->ValidityDomain, param_name);
	  print_evalue(stdout,&en->EP, param_name);
	  printf( "\n-----------------------------------\n" );
	}

	return( 0 );
}



