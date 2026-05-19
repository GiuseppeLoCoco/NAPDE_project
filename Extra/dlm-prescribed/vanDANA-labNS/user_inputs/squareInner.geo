h = 0.2;

Point(11) = {-0.5, 0, 0, h};
Point(12) = {0, -0.5, 0, h};
Point(13) = {0.5, 0, 0, h};
Point(14) = {0, 0.5, 0, h};
Line(11) = {11,12};
Line(12) = {12,13};
Line(13) = {13,14};
Line(14) = {14,11};
Line Loop(20) = {11,12,13,14};
Surface(2) = {20};
Physical Surface(2) = {2};
Physical Curve(20) = {11,12,13,14};
