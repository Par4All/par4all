// Ticket 550

// The const qualifier is added to the typedef JOCTET instead of being
// added to the type of next_input_byte. This shows up when the parsed
// code is printed: the typedef declaration for JOCTET is wrong.

typedef unsigned char JOCTET;

struct jpeg_source_mgr {
  const JOCTET * next_input_byte;
};
