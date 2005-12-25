/*
 * $Id$
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
