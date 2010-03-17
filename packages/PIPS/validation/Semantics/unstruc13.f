      program unstruc13

C     Excerpt from calao.f in SemanticsPrivate

      real z(10)

      do 41 k=iinf,isup
         if (z(k).gt.zmoyen) go to 42
 41   continue
 42   volume=zmoyen

      end
