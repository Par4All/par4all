      program unstruc02

C     Check bug in controlizer when controlize_test() is called from controlize_list()

      real p(10,10,10), a0(10,10,10)

      do 31 i=1,nim1
         do 31 j=1,njm1
            if(p(i,j,km).eq.0.) ao(i,j,1)=0.
            if(p(i,j,km).eq.0.) goto 31
            enbas=0.
 31   continue

      end
