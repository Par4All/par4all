/*
 * $Id$ 
 */

package fr.ensmp.cri.jpips;

import fr.ensmp.cri.jpips.Pawt.*;
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
