      integer function absval(n)
      if(n.lt.0) then
         absval = -n
      else
         absval = n
      endif
      return
      end
