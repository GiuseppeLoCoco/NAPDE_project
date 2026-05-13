SetFactory("OpenCASCADE");
h = 1.0/64.0;

Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(1) = {1,2,3,4};

Point(11) = {0.75, 0.5, 0, h};
Point(12) = {0.5, 0.625, 0, h};
Point(13) = {0.25, 0.5, 0, h};
Point(14) = {0.5, 0.375, 0, h};
Point(15) = {0.5, 0.5, 0, h};
Ellipse(11) = {11,15,11,12};
Ellipse(12) = {12,15,11,13};
Ellipse(13) = {13,15,11,14};
Ellipse(14) = {14,15,11,11};
Line Loop(10) = {11,12,13,14};

Point(31) = {0.85, 0.5, 0, h};
Point(32) = {0.5, 0.7, 0, h};
Point(33) = {0.15, 0.5, 0, h};
Point(34) = {0.5, 0.3, 0, h};
Ellipse(31) = {31,15,31,32};
Ellipse(32) = {32,15,31,33};
Ellipse(33) = {33,15,31,34};
Ellipse(34) = {34,15,31,31};
Line Loop(30) = {31,32,33,34};

Line(311) = {32, 12};
Line(331) = {34, 14};
Line Loop(300) = {31, 311, -11, -14, -331, 34};

Plane Surface(1) = {10};
Plane Surface(10) = {1,30};
Plane Surface(30) = {30,10};
Plane Surface(300) = {300};
Plane Surface(301) = {1,300};

//BooleanDifference{Surface{1};Delete;}{Surface{10};Delete;}
//
//Point(11) = {0.75, 0.5, 0, h};
//Point(12) = {0.5, 0.625, 0, h};
//Point(13) = {0.25, 0.5, 0, h};
//Point(14) = {0.5, 0.375, 0, h};
//Point(180) = {0.5, 0.5, 0, h};
//Ellipse(11) = {11,180,11,12};
//Ellipse(12) = {12,180,11,13};
//Ellipse(13) = {13,180,11,14};
//Ellipse(14) = {14,180,11,11};
//Line Loop(10) = {11,12,13,14};
//Plane Surface(50) = {10};

Physical Surface(1) = {1};
Physical Surface(10) = {10};
Physical Surface(30) = {30};
Physical Surface(300) = {300};
Physical Surface(301) = {301};
//Physical Curve(20) = {11,12,13,14};
//Physical Curve(10) ={1,2,3,4};

Mesh.MeshOnlyVisible = 1;