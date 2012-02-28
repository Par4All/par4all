!     Floating point sizing

!     http://svn.cri.ensmp.fr/trac/pips/ticket/678
      
      program main
      
      integer*4 i
      parameter ( i = 100 )
      integer*4 j
      parameter ( j = i*0.1 )

      integer*4 array

      common / C1 / 
     *     array(j)

      array(1) = 2
      print *,array(1)

      end
