This program use trained haar-cascade classifier to detect face.
To improve the efficiency of the detection of the face.I restrict the area for face detection 
in the neighborhood of the face detect location in the previous frame but not the full frame.
However, I add an other control that if in the sub rectangle(the neighborhood of the face detect 
location in the previous frame) the program can't find any faces for some time, it will start to 
detect face in the full frame again.I set the threshold to 3000ms.