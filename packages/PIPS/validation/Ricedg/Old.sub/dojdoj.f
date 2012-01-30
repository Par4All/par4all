c     Test pour la reutilisation des indices des boucles
      program dojdoj
      real *8 T(11,11)

      do 10 i = 1, 10
         do 20 j = 1,5
            T(i,j) = T(i-1, j-1)
 20      continue

          do 30 j = 6, 10
             T(i+1,j+1) = T(i,j)
 30       continue
 10    continue
       end
