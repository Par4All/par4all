/*
 * $Id$
 * 
 * $Log: Pawt.java,v $
 * Revision 1.7  1998/10/17 09:50:08  coelho
 * white background color.
 *
 * Revision 1.6  1998/10/16 16:53:06  coelho
 * swing package updated.
 *
 * Revision 1.5  1998/10/16 13:55:59  coelho
 * import fixed.
 *
 * Revision 1.4  1998/07/03 16:36:51  coelho
 * PComboBox includes a direct field.
 *
 * Revision 1.3  1998/07/03 08:13:33  coelho
 * simpler constructors. tips added to labels and buttons.
 *
 * Revision 1.2  1998/06/30 17:35:33  coelho
 * abstarct wt for jpips.
 *
 * Revision 1.1  1998/06/30 16:40:14  coelho
 * Initial revision
 *
 */

package JPips;

import java.util.*;

import java.awt.*;
import java.awt.event.*;

import com.sun.java.swing.*;
import com.sun.java.swing.event.*;

/** A graphical package for JPips
  * All components extends from swing components.
  * Some components contain informations to manage their Events.
  * 
  * @author Francois Didry
  */
interface Pawt
{
  class PComponent extends com.sun.java.swing.JComponent
  { 
    public String checking; 
    PComponent() { super(); } 
  }
  
  class PFrame extends com.sun.java.swing.JFrame
  { 
    public Vector optionVector;
    PFrame(String name){ super(name); }
    
    /** Disables or enables the Option objects of JPips.
     * @param yes true means disable
     */
    public void lock(boolean yes)
    {
      for(int i=1; i<optionVector.size(); i++)
	((Activatable) optionVector.elementAt(i)).setActivated(!yes);
    }
  }

  class PDialog extends com.sun.java.swing.JDialog
    { PDialog(Frame f, String s, boolean b){ super(f, s, b); } }
  
  class PPanel extends com.sun.java.swing.JPanel
  {
    PPanel(){ super(); }
    PPanel(LayoutManager l){ super(l); }
  }
  
  class PMenuBar extends com.sun.java.swing.JMenuBar
  { PMenuBar(){ super(); } }
  
  class PMenu extends com.sun.java.swing.JMenu
  { PMenu(){ super(); }
    PMenu(String name){ super(name); }
  }
  
  class PMenuItem extends com.sun.java.swing.JMenuItem
  {
    public String command;
    public PTextField tf;
    PMenuItem(String name){ super(name); }
    PMenuItem(String name,String command)
    { super(name); this.command = command; }
    PMenuItem(String name,String command,PTextField tf)
    { super(name); this.command = command; this.tf = tf; }
  }
  
  class PCheckBox 
    extends com.sun.java.swing.JCheckBox
  {
    public String command,checking;
    public PCheckBoxMenuItem cbmi;
    public PTextFrame frame;
    
    PCheckBox(String name, String command, String checking, String tip, 
	      PCheckBoxMenuItem cbmi, PTextFrame frame)
    { 
      super(name); 
      if (tip!=null) setToolTipText(tip);
      this.command = command; 
      this.checking = checking;
      this.cbmi = cbmi; 
      this.frame = frame;
    }	
    
    PCheckBox(String name)
    { this(name, null, null, null, null, null); }
    
    PCheckBox(String name, String command, String checking, String tip)
    { this(name, command, checking, tip, null, null); }
    
    PCheckBox(String name, PTextFrame frame)
    { this(name, null, null, null, null, frame); }
  }
  
  class PCheckBoxMenuItem extends com.sun.java.swing.JCheckBoxMenuItem
  {
    public String command,checking;
    public PCheckBox cb;
    
    PCheckBoxMenuItem(String name){ super(name); }
    
    PCheckBoxMenuItem(String name,String command,String checking)
    { 
      super(name);
      this.command = command;
      this.checking = checking; 
    }
    
    PCheckBoxMenuItem(String name, String command, 
		      String checking, PCheckBox cb)
    {
      super(name); 
      this.command = command; 
      this.checking = checking;
      this.cb = cb; 
    }
  }

  class PRadioButtonMenuItem 
    extends com.sun.java.swing.JRadioButtonMenuItem
  {
    public String command, checking;
    public PLabel label;
    public PComboBox cob;
    public Object o;
    public PRadioButtonMenuItem rbmi;

    PRadioButtonMenuItem(String name, String command, 
			 PComboBox cob, Object o, String checking)
    { super(name); this.command = command; this.cob = cob; this.o = o;
    this.checking = checking; }
    
    PRadioButtonMenuItem(String name){ super(name); }
    PRadioButtonMenuItem(String name, String command)
    { super(name); this.command = command; }
    PRadioButtonMenuItem(String name, String command, PLabel label,
			 PRadioButtonMenuItem rbmi)
    { super(name); this.command = command; this.label = label;
    this.rbmi = rbmi; }
    PRadioButtonMenuItem(String name, String command, PLabel label,
			 PRadioButtonMenuItem rbmi, String checking)
    { super(name); this.command = command; this.label = label;
	  this.rbmi = rbmi; this.checking = checking; }
    PRadioButtonMenuItem(String name, String command,
			 PComboBox cob, Object o)
    { super(name); this.command = command; this.cob = cob; this.o = o; }
  }


  class PSeparator 
    extends com.sun.java.swing.JSeparator
  { 
    PSeparator(){ super(); } 
  }

  class PButton 
    extends com.sun.java.swing.JButton
  {
    public String 	command;
    public PTextField 	tf;
    public PTextFrame 	frame;

    PButton(String name, String command, String tip, 
	    PTextField tf, PTextFrame frame)
      { 
	super(name); 
	setMargin(new Insets(0,0,0,0));
	if (tip!=null) setToolTipText(tip);
	this.command = command;
	this.tf = tf; 
	this.frame = frame;
      }

    PButton(String name) 
      { this(name, null, null, null, null); }
    PButton(String name, PTextFrame frame) 
      { this(name, null, null, null, frame); }
    PButton(String name, String command) 
      { this(name, command, null, null, null); }
    PButton(String name, String command, String tip)
      {	this(name, command, tip, null, null); }
  }

  class PLabel 
    extends com.sun.java.swing.JLabel
  {
    PLabel(String s, String tip)
      { 
	super(s); 
	if (tip!=null) setToolTipText(tip);
      }

    PLabel(String s) { this(s, null); }
    PLabel() { super(); }
  }
    
  class PTextField extends com.sun.java.swing.JTextField
  {
    PTextField(String s)
    {
      super(s); 
      setBackground(Color.white); 
    }

    PTextField()
    { 
      this(""); 
    }
  }
    
  class PList extends com.sun.java.swing.JList
  {
    PList() { super(); }
    PList(Vector v) { super(v); }
    PList(DefaultListModel l) { super(l); }
  }

  /** A PComboBox can send directly its command on selections,
    * or wait for some specific action to do the job (e.g. a button pressed).
    */
  class PComboBox extends com.sun.java.swing.JComboBox
  {
    public String checking, marker;
    public boolean direct;
    public Vector vCommand = new Vector(),
      vRbmi = new Vector(),
      vChecking = new Vector();
    
    PComboBox(String ch, String ma, boolean direct)
    { 
      super(); 
      checking = ch; 
      marker = ma; 
      this.direct = direct;
    }
    
    PComboBox(){ this(null, null, true); }
    PComboBox(String ch){ this(ch, null, true); }
    PComboBox(String ch, String ma) { this(ch, ma, true); }
  }

  class PScrollPanel extends com.sun.java.swing.JScrollPane
  {
    PScrollPanel(){ super(); } 
    PScrollPanel(Component c){ super(c); } 
  }
        
  class PButtonGroup extends com.sun.java.swing.ButtonGroup
  {
    public String checking;
    PButtonGroup(String ch){ super(); checking = ch; }
  }
        
  class PTextArea extends java.awt.TextArea
  {
    PTextArea(String s)
    { 
      super(s);  
      setBackground(Color.white); 
    }
    
    PTextArea()
    { 
      this("");
    }
  }
        
  class PTextFrame extends com.sun.java.swing.JFrame
  {
    boolean locked;
    boolean writable;
    PTextArea ta;
    PButton panelButton;
    
    PTextFrame(String name, String text, boolean locked, boolean writable)
    {
      //window
      super(name);
      this.locked = locked;
      this.writable = writable;
      PButton b;
      getContentPane().setLayout(new BorderLayout());
      ta = new PTextArea(text);
      ta.setFont(new Font("Monospaced", Font.PLAIN, 12));
      ta.setEnabled(writable);
      PScrollPanel s = new PScrollPanel((Component)ta);
      getContentPane().add(s, BorderLayout.CENTER);
      
      PPanel p = new PPanel(new GridLayout(1,3));
      PCheckBox cb = new PCheckBox("Locked",this);
      cb.setSelected(locked);
      cb.addActionListener(new ActionListener() {
	public void actionPerformed(ActionEvent e) {
	  PCheckBox check = (PCheckBox)e.getSource();
	  check.frame.locked = check.isSelected();
	}
      });
      p.add(cb);
      b = new PButton("Hide",this);
      b.addActionListener(new ActionListener() {
	public void actionPerformed(ActionEvent e) {
	  PButton button = (PButton)e.getSource();
	  button.frame.setVisible(false);
	}
      });
      p.add(b);
      b = new PButton("Close",this);
      b.addActionListener(new ActionListener() {
	public void actionPerformed(ActionEvent e) {
	  PButton button = (PButton)e.getSource();
	  button.frame.dispose();
	}
      });
      p.add(b);
      getContentPane().add(p, BorderLayout.SOUTH);
      
      //jpips button
      panelButton = new PButton(name,this);
      panelButton.addActionListener(new ActionListener() {
	public void actionPerformed(ActionEvent e) {
	  PButton button = (PButton)e.getSource();
	  button.frame.setVisible(true);
	  button.frame.toFront();
	}
      });
      
      pack();
      setSize(new Dimension(600,400));
    }
  }
}
