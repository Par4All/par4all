c
c Fake sources for PIPS to deal with special FC directives
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
