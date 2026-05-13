h = 0.2;

Point(1) = {-1, -1, 0, h};
Point(2) = {1, -1, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {-1, 1, 0, h};
Point(5) = {0, 0, 0, h};
Circle(1) = {1,5,2};
Circle(2) = {2,5,3};
Circle(3) = {3,5,4};
Circle(4) = {4,5,1};
Line Loop(10) = {1,2,3,4};
Surface(1) = {10};
Physical Surface(1) = {1};
Physical Curve(10) = {1,2,3,4};
