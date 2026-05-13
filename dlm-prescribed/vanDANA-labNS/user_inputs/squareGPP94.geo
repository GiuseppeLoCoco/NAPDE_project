h = 1.0/32.0;

Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(10) = {1,2,3,4};
Surface(1) = {10};
Physical Surface(1) = {1};
Physical Curve(10) = {1,2,3,4};
