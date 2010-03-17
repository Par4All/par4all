      subroutine altret01(x, *)

      data iy /1/

C     Check different handlings of alternate return constructs

      call foo(*123)

      call bar(x, *123)

      call bar2(*123, x)

      if(x.gt.float(iy)) return

 123  continue
      return 1

      end
