/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

package fr.ensmp.cri.jpips;

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
