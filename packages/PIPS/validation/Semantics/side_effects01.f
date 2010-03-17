      program side_effects01

      i = 2
      j = i + inc(i)
      k = inc(i) + i
      print *, j, k
      if(i.eq.inc(i)) then
         print *, "An aggressive optimization option is used"
      else
         print *, "An standard optimization option is used"
      endif

      end
      function inc(k)
      k = k + 1
      inc = k
      end
