/*
 * $Id$ 
 *
 * $Log: JPipsComponent.java,v $
 * Revision 1.3  1998/06/30 17:35:33  coelho
 * a graphical and logical component in jpips.
 *
 * Revision 1.2  1998/06/30 16:22:26  coelho
 * cleaner (wrt my standards).
 *
 * Revision 1.1  1998/06/09 07:28:28  didry
 * Initial revision
 */

package JPips;

import JPips.Pawt.*;
import java.awt.*;

/** An interface that defines methods for JPips components.
  *
  * @author Francois Didry
  */
public interface JPipsComponent
  extends Activatable
{
  public Component getComponent();
  public PMenu getMenu();
  public void reset();
}
