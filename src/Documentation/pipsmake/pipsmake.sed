s,	,    ,g;
s,  *$,,;
/^\\begin{PipsMake}/,/^\\end{PipsMake}/!d;
/^\\begin{PipsMake}/d;
/^\\end{PipsMake}/s,.*,,
