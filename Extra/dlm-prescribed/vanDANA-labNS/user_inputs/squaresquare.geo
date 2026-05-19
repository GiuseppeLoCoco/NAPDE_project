h = 0.1;

Point(1) = {-1, -1, h};
Point(2) = {1, -1, h};
Point(3) = {1, 1, h};
Point(4) = {-1, 1, h};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(10) = {1,2,3,4};
Surface(1) = {10};
Physical Surface(1) = {1};
Physical Curve(10) = {1,2,3,4};

Point(11) = {-0.5, 0, h};
Point(12) = {0, -0.5, h};
Point(13) = {0.5, 0, h};
Point(14) = {0, 0.5, h};
Line(11) = {11,12};
Line(12) = {12,13};
Line(13) = {13,14};
Line(14) = {14,11};
Line Loop(20) = {11,12,13,14};
Surface(2) = {20};
Physical Surface(2) = {2};
Physical Curve(20) = {11,12,13,14};
