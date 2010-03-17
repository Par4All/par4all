! parameter dependences 
      program decl6
      integer a, b, c, d, e, o, p, n, m, x, y, z, u
      parameter (a=1, b=10, c=12, d=14, e=15)
      parameter (o=a+4, p=o, n=o+p, m=n+5, u=120)
      parameter (x=2, y=3, z=y+x)
      integer a1,a2,a3,a4,a5,a6,a7,a8
      parameter (a1=1,a2=a1,a3=a2,a4=a3,a5=a4,a8=8,a7=a8,a6=a7)
      parameter(i1=1,i2=i1,i3=i2,i4=i3,i5=i4,i6=i5)
      integer t(m), i
      do i=1, 5
         t(i) = b+c+d+e+z+i6
      enddo
      print *, a5, a7
      end
