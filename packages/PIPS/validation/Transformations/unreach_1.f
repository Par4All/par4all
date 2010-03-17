! basic detection of unreachable code...
      program u1
      print *, 'hello'
      call unst(2.21)
      call foo(0.0)
      print *, 'dead, since foo stops'
      STOP
      call foo(1.1)
      print *, 'dead'
      end

      subroutine foo(x)
      if (x.le.2.1) then
         print *, 'bonjour'
         STOP
      else
         print *, 'hello'
         call noend
      endif
      print *, 'also dead'
      end
      
      subroutine unst(x)
      real x, y
      y = 12.345
      print *, 'hello'
      if (x.gt.0.0) then
 10   print *, 'live'
      STOP
      print *, 'dead'
      if (x.lt.y) goto 20
      print *, 'dead 2'
 20   goto 10
      print *, 'graved'
      endif
      print *, 'live 2'
      end

      subroutine noend
 1    goto 1
      print *, 'dead in noend'
      end
