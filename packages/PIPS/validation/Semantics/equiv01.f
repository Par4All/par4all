      program equiv01

C     Check that unusable information is not used

      equivalence (i,j), (k,x)

      read*, i, j, k, x

      if(i.gt.n) then
         i = i - 1
         print *, i
      endif

      if(j.gt.n) then
         j = j - 1
         print *, j
      endif

      if(k.gt.n) then
         k = k - 1
         print *, k
      endif

      print *, i, j, k, x

      end
