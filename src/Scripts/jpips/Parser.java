/*
 * $Id$
 *
 * $Log: Parser.java,v $
 * Revision 1.3  1998/07/02 18:31:24  coelho
 * simpler parser.
 *
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
    * basically it is readLine()... (FC)
    */
  public Parser(FileReader f)
    {
      super(f);
      resetSyntax();

      eolIsSignificant(false);

      wordChars(' ','~'); // from 32 to 126

      whitespaceChars(this.TT_EOL,this.TT_EOL);      
      whitespaceChars('\t','\t');      

      commentChar('#');
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

      // System.err.println("ret: " + this.sval);
      return this.sval;
    }
}
