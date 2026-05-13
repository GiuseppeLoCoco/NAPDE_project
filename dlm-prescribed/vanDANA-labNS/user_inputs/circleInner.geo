h = 0.02;
raggio = 1;

Point(11) = {-raggio, 0, 0, h};
Point(12) = {0, -raggio, 0, h};
Point(13) = {raggio, 0, 0, h};
Point(14) = {0, raggio, 0, h};
Point(15) = {0, 0, 0, h};
Circle(11) = {11,15,12};
Circle(12) = {12,15,13};
Circle(13) = {13,15,14};
Circle(14) = {14,15,11};
Line Loop(20) = {11,12,13,14};
Surface(2) = {20};
Physical Surface(2) = {2};
Physical Curve(20) = {11,12,13,14};
