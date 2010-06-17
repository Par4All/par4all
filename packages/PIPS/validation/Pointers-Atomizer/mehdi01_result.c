Proper effects for "main"

int main()
{
   struct p_s {
      struct q_s *q;
   };
   struct q_s {
      int r;
   };
   struct p_s *p;
   struct q_s *b;
   struct q_s *u;
   struct q_s *x;
   struct q_s **z;
//               <must be written>: a
   int a = 0;
//               <must be read   >: _MALLOC_EFFECTS:_MALLOC_
//               <must be written>: _MALLOC_EFFECTS:_MALLOC_ u
   u = (struct q_s *) malloc(sizeof(struct q_s ));
//               <must be read   >: _MALLOC_EFFECTS:_MALLOC_
//               <must be written>: _MALLOC_EFFECTS:_MALLOC_ b
   b = (struct q_s *) malloc(sizeof(struct q_s ));
//               <must be read   >: _MALLOC_EFFECTS:_MALLOC_
//               <must be written>: _MALLOC_EFFECTS:_MALLOC_ p
   p = (struct p_s *) malloc(sizeof(struct p_s ));
//               <must be read   >: b
//               <must be written>: x
   x = b;
//               <may be written >: *ANY_MODULE*:*ANYWHERE*
//               <must be read   >: p x
   p->q = x;
//               <must be read   >: p
//               <must be written>: z
   
   
   
   z = &p->q;
//               <may be written >: *ANY_MODULE*:*ANYWHERE*
//               <must be read   >: b p
   p->q = b;
//               <must be written>: z
   z = &u;
//               <may be read    >: *ANY_MODULE*:*ANYWHERE*
//               <may be written >: *ANY_MODULE*:*ANYWHERE*

   p->q->r = a;
   return 0;
}

POINTS TO for "main"

//  {}
int main()
{
//  points to = {}
   struct p_s {
      struct q_s *q;
   };
//  points to = {}
   struct q_s {
      int r;
   };
//  points to = {}
   struct p_s *p;
//  points to = {}
   struct q_s *b;
//  points to = {}
   struct q_s *u;
//  points to = {}
   struct q_s *x;
//  points to = {}
   struct q_s **z;
//  points to = {}
   int a = 0;
//  points to = {}
   u = (struct q_s *) malloc(sizeof(struct q_s ));
//  {(u,*HEAP*_l_20,-Exact-)}
   b = (struct q_s *) malloc(sizeof(struct q_s ));
//  {(b,*HEAP*_l_21,-Exact-);(u,*HEAP*_l_20,-Exact-)}
   p = (struct p_s *) malloc(sizeof(struct p_s ));
//  {(b,*HEAP*_l_21,-Exact-);(p,*HEAP*_l_22,-Exact-);(u,*HEAP*_l_20,
//   -Exact-)}
   x = b;
//  {(b,*HEAP*_l_21,-Exact-);(p,*HEAP*_l_22,-Exact-);(u,*HEAP*_l_20,
//   -Exact-);(x,*HEAP*_l_21,-Exact-)}
   p->q = x;
//  {(b,*HEAP*_l_21,-Exact-);(p[0][q],*HEAP*_l_21,-Exact-);(p,
//   *HEAP*_l_22,-Exact-);(u,*HEAP*_l_20,-Exact-);(x,*HEAP*_l_21,
//   -Exact-)}
   
   
   
   z = &p->q;
//  {(b,*HEAP*_l_21,-Exact-);(p[0][q],*HEAP*_l_21,-Exact-);(p,
//   *HEAP*_l_22,-Exact-);(u,*HEAP*_l_20,-Exact-);(x,*HEAP*_l_21,
//   -Exact-);(z,p[0][q],-Exact-)}
   p->q = b;
//  {(b,*HEAP*_l_21,-Exact-);(p[0][q],*HEAP*_l_21,-Exact-);(p,
//   *HEAP*_l_22,-Exact-);(u,*HEAP*_l_20,-Exact-);(x,*HEAP*_l_21,
//   -Exact-);(z,p[0][q],-Exact-)}
   z = &u;
//  {(b,*HEAP*_l_21,-Exact-);(p[0][q],*HEAP*_l_21,-Exact-);(p,
//   *HEAP*_l_22,-Exact-);(u,*HEAP*_l_20,-Exact-);(x,*HEAP*_l_21,
//   -Exact-);(z,u,-Exact-)}

   p->q->r = a;
//  {(b,*HEAP*_l_21,-Exact-);(p[0][q],*HEAP*_l_21,-Exact-);(p,
//   *HEAP*_l_22,-Exact-);(u,*HEAP*_l_20,-Exact-);(x,*HEAP*_l_21,
//   -Exact-);(z,u,-Exact-)}
   return 0;
}

Proper effects using points_to

int main()
{
   struct p_s {
      struct q_s *q;
   };
   struct q_s {
      int r;
   };
   struct p_s *p;
   struct q_s *b;
   struct q_s *u;
   struct q_s *x;
   struct q_s **z;
//               <must be written>: a
   int a = 0;
//               <must be read   >: _MALLOC_EFFECTS:_MALLOC_
//               <must be written>: _MALLOC_EFFECTS:_MALLOC_ u
   u = (struct q_s *) malloc(sizeof(struct q_s ));
//               <must be read   >: _MALLOC_EFFECTS:_MALLOC_
//               <must be written>: _MALLOC_EFFECTS:_MALLOC_ b
   b = (struct q_s *) malloc(sizeof(struct q_s ));
//               <must be read   >: _MALLOC_EFFECTS:_MALLOC_
//               <must be written>: _MALLOC_EFFECTS:_MALLOC_ p
   p = (struct p_s *) malloc(sizeof(struct p_s ));
//               <must be read   >: b
//               <must be written>: x
   x = b;
//               <must be read   >: p x
//               <must be written>: *HEAP*_l_22[.q]
   p->q = x;
//               <must be read   >: p
//               <must be written>: z
   
   
   
   z = &p->q;
//               <must be read   >: b p
//               <must be written>: *HEAP*_l_22[.q]
   p->q = b;
//               <must be written>: z
   z = &u;
//               <must be read   >: *HEAP*_l_22[.q] a p
//               <must be written>: *HEAP*_l_21[.r]

   p->q->r = a;
   return 0;
}

