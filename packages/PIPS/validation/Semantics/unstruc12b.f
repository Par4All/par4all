      program unstruc12b

C     Reduced version of unstruc12 to track a bug in unstructured

      real p(10,10,10), z(10), ao(10,10,10)

         do 31 j=1,njm1
            do 41 k=iinf,isup
               if (z(k).gt.zmoyen) go to 42
 41         continue
 42         volume=zmoyen
            if (somme.eq.0.) goto 44
 31      continue
 44      if(impc.eq.1) write(6,*) somme

         end
