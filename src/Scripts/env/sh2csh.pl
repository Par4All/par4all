#! /usr/bin/perl -wn

if (/^test/ or /^}/ or /^\#/ or /^\s*$/) 
{
    print;
}
elsif (/^\s*export\s+([A-Za-z_0-9]+)=(.*)/)
{
    print "setenv $1 $2\n";
}
elsif (/^\s*([A-Za-z_0-9]+)=(.*)/)
{
    # handle PATH especially for csh
    if ($1 eq 'PATH')
    {
	print "setenv $1 $2\n";
    }
    else
    {
        print "set $1=$2\n";
    }
}

END 
{ 
    print "rehash\n"; 
}
