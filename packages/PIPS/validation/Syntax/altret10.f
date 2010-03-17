      program altret10

C     Check link-edit issues

      i = 0

      call bar(*123, i)
      print *, "First standard return from bar"

      call bar(*123, i)
C     This should not happen!
      print *, "Second standard return from bar"

      print *, 'STOP 0 in ALTRET10'
      stop 0

 123  continue
      print *, 'STOP 1 in ALTRET10'
      stop 1

      end

      subroutine bar(*, i)
      call foo(*256, i)
      return
 256  return 1
      end

      subroutine foo(*, i)
      print *, "foo is entered with ", i
      if(i.gt.0) then
         print *, 'RETURN 1 in FOO'
         return 1
      endif
      i = i + 1
      end
