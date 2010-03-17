      program altret08

C     Check link-edit issues

      i = 0

      call foo(*123, i)
      print *, "First standard return from foo"

      call foo(*123, i)
C     This should not happen!
      print *, "Second standard return from foo"

      print *, 'STOP 0 in ALTRET08'
      stop 0

 123  continue
      print *, 'STOP 1 in ALTRET08'
      stop 1

      end

      subroutine foo(*, i)
      print *, "foo is entered with ", i
      if(i.gt.0) then
         print *, 'RETURN 1 in FOO'
         return 1
      endif
      i = i + 1
      end
