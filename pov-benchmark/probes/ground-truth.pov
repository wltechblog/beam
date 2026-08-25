// Ground-truth probe: exact benchmark camera + the 3 problem objects.
// Texts colored red/green so pixel analysis can find them; logo purple.
#version 3.7;
global_settings { assumed_gamma 1.0 }

camera {
   location  <3.2, 3.2, 1.8>
   direction y
   sky       z
   up        z
   right     -x*(image_width/image_height)
   look_at   <-1, -1, 0.9>
   angle     45
}

light_source {
   <-0.7, 0.83, 0.24>*150000
   color rgb <3.43,2.87,1.95>
}

// basin wall behind the texts (stone-grey so we see placement context)
difference {
   box { <-2.0, -1.7, 0.5>, < 2.0, 1.7, -6> }
   box { <-1.9, -1.6, 1.0>, < 1.9, 1.6, -5> }
   pigment { rgb <0.55, 0.52, 0.5> }
}

// ground
plane { z, -0.3 pigment { rgb <0.5, 0.45, 0.4> } }

// ---- the two texts, verbatim benchmark chains, colored ----
#declare POV_Text =
text {
   internal 1
   "POV-Ray"
   0.25, 0
   scale 0.3
   rotate 90*x
   rotate -90*z
   pigment { rgb <1, 0.1, 0.1> }
}

#declare Version_Text =
text {
   internal 1
   "Version 3.8"
   0.25, 0
   scale 0.3
   rotate 90*x
   rotate -90*z
   pigment { rgb <0.1, 1, 0.1> }
}

object { POV_Text translate <-1.97, 0.995, 0.575> }
object { Version_Text rotate 90*z translate <-0.5, -1.5, 0.575> }

// ---- the logo, verbatim benchmark chain ----
#declare Povray_Logo =
merge {
   sphere {2*y, 1}
   difference {
      cone {2*y, 1, -4*y, 0}
      sphere {2*y, 1.4 scale <1,1,2>}
   }
   difference {
      sphere {0, 1 scale <2.6, 2.2, 1>}
      sphere {0, 1 scale <2.3, 1.8, 2> translate <-0.35, 0, 0>}
      rotate z*30 translate 2*y
   }
   rotate <0, 0, -25>
   translate <-0.5,-0.35,0>
   scale 1/4
}

object {
   Povray_Logo
   rotate 90*x
   scale 0.4
   translate -0.2*y
   pigment { rgb <0.65, 0.55, 0.9> }
   finish { ambient 0.25 diffuse 0.7 }
   rotate -25*z
   translate 0.96*z
   translate <0.0, 0.0, 0.6>
}

// ---- axis markers for empirical orientation ----
sphere { <0,0,0>, 0.08 pigment { rgb <1,1,0> } }          // yellow: origin
sphere { <1,0,0>, 0.08 pigment { rgb <0,1,1> } }          // cyan: +x
sphere { <0,1,0>, 0.08 pigment { rgb <1,0,1> } }          // magenta: +y
sphere { <0,0,1>, 0.08 pigment { rgb <0.2,0.4,1> } }      // blue: +z
