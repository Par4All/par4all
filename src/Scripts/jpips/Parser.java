/*
 * $Id$
 *
 * $Log: Parser.java,v $
 * Revision 1.2  1998/07/01 07:06:22  coelho
 * cleaner.
 *
 * Revision 1.1  1998/06/30 17:35:33  coelho
 * Initial revision
 */

package JPips;

import java.lang.*;
import java.util.*;
import java.io.*;

/** A sub class of StreamTokenizer which parses line by line.
  * 
  * @author Francois Didry
  */
public class Parser 
    extends StreamTokenizer
{
  /** Defines a StreamTokenizer.
    */
  public Parser(FileReader f)
    {
      super(f);
      
      wordChars('0', '0');
      wordChars('1', '1');
      wordChars('2', '2');
      wordChars('3', '3');
      wordChars('4', '4');
      wordChars('5', '5');
      wordChars('6', '6');
      wordChars('7', '7');
      wordChars('8', '8');
      wordChars('9', '9');
      wordChars('_', '_');
      wordChars('<', '<');
      wordChars('>', '>');
      wordChars('[', '[');
      wordChars(']', ']');
      wordChars('/', '/');
      wordChars('.', '.');
      wordChars('+', '+');
      wordChars('-', '-');
      wordChars('(', '(');
      wordChars(')', ')');
      wordChars(':', ':');
      wordChars(' ', ' ');
      wordChars('!', '!');
      wordChars('@', '@');
      wordChars('$', '$');
      wordChars('%', '%');
      wordChars('^', '^');
      wordChars('&', '&');
      wordChars('*', '*');
      wordChars(';', ';');
      wordChars('\'', '\'');
      wordChars('`', '`');
      wordChars('~', '~');
      wordChars('?', '?');
      wordChars('/', '/');
      wordChars('\\', '\\');
      wordChars('|', '|');
      wordChars('.', '.');
      wordChars(',', ',');
      wordChars('"', '"');
 
      commentChar('#');
      eolIsSignificant(false);
      whitespaceChars(this.TT_EOL,this.TT_EOL);      
      whitespaceChars('\t','\t');      
    }
    
  /** @return the next line which is not null or null if EOF is reached
    */
  public String nextNonEmptyLine()
    {
      try
        {
          while(this.nextToken() != this.TT_EOF && this.sval == null);
	}
      catch(IOException e)
        {
	  System.out.println(e);
	  System.out.flush();
	}
      return this.sval;
    }
}
