c
c Fake sources for PIPS to deal with special FC directives. 
c functions with side effects, just in case. 
c
c (c) Fabien COELHO, 09/95
c
c $RCSfile: hpfc_stubs.f,v $ (version $Revision$)
c ($Date: 1996/04/17 17:44:54 $, )
c     
c synchronization 
      subroutine hpfc1()
      print *, 'hpfc1: '
      end
c timer on
      subroutine hpfc2()
      print *, 'hpfc2: '
      end
c timer off
      subroutine hpfc3(comment)
      character comment*(*)
      print *, 'hpfc3: ', comment
      end
c io/host section marker
      subroutine hpfc7()
      print *, 'hpfc7: '
      end
c dead FC directive. one argument, why not...
      subroutine hpfc8(x)
      integer x
      print *, 'hpfc8: ', x
      end
c
c That is all
c
