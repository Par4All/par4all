      program unstruc12

C     Excerpt from calao.f in SemanticsPrivate

      real p(10,10,10), z(10), ao(10,10,10)

      do 31 i=1,nim1
         do 31 j=1,njm1
            if(p(i,j,km).eq.0.) goto 31
            do 41 k=iinf,isup
               if (z(k).gt.zmoyen) go to 42
 41         continue
 42         volume=zmoyen
            if (somme.eq.0.) goto 44
            if (abs(diffe).ge.abs(somme)) goto 45
 31      continue
         goto 60
 44      if(impc.eq.1) write(6,*) somme
         goto 60
 45      if(impc.eq.1) write(6,*)
         goto 60
 66      if(impc.eq.1) write(6,*) ao(im1,jm1,1)
 60      continue

         end
