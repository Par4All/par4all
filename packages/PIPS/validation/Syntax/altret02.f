      subroutine altret02(x, *, *)

      call foo(*123)

      call bar(x, *123)

      call bar2(*123, x)

      call bar3(*123, x, *234)

      if(x.gt.0.) return

 123  continue
      return 1

 234  return 2

      end

      subroutine foo(*)

      print *, "foo was called"

      end

      subroutine bar(x, *)

      return 1

      end

      subroutine bar2(*, x)

      return 1

      end

      subroutine bar3(*, x, *)

      return 2

      end
