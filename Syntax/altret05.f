      program altret05

C     Check link-edit issues

      i = 0

      call foo(*123, i)

      call foo(*123, i)

      stop 0

 123  continue
      stop 1

      end

      subroutine foo(*, i)
      print *, "foo is entered with ", i
      if(i.gt.0) return 1
      i = i + 1
      end
