c
c Fake sources for PIPS to deal with special FC directives. 
c functions with side effects, just in case. 
c
c (c) Fabien COELHO, 09/95
c
c $RCSfile: hpfc_stubs.f,v $ (version $Revision$)
c ($Date: 1995/09/05 13:41:45 $, )
c      
      subroutine hpfc1()
      print *, 'hpf1: '
      end
      subroutine hpfc2()
      print *, 'hpf2: '
      end
      subroutine hpfc3(comment)
      character comment*(*)
      print *, 'hpf3: ', comment
      end
c
c That is all
c
