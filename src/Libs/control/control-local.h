/* -- control.h */

#ifndef CONTROL_INCLUDED
#define CONTROL_INCLUDED

static void get_blocs( c, l )
control c ;
cons **l ;
{
    MAPL( cs, {if( CONTROL( CAR( cs )) == c ) return ;}, *l ) ;
    *l = CONS( CONTROL, c, *l ) ;
    MAPL( cs, {get_blocs( CONTROL( CAR( cs )), l );}, control_successors( c )) ;
    MAPL( ps, {get_blocs( CONTROL( CAR( ps )), l );}, control_predecessors( c )) ;
}

#define CONTROL_MAP( ctl, code, c, list ) \
{ \
    cons *_cm_list_init = (list) ; \
    cons *_cm_list = _cm_list_init ; \
    if( _cm_list == NIL ) {\
         get_blocs( c, &_cm_list ) ; \
         _cm_list = gen_nreverse( _cm_list ) ; \
    }\
    MAPL( _cm_ctls, {control ctl = CONTROL( CAR( _cm_ctls )) ; \
 \
		 code ;}, \
	  _cm_list ) ; \
   if( _cm_list_init == NIL ) \
        list = _cm_list ; \
}

#endif
