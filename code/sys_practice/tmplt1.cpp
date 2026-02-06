template <typename T>
T minimum(const T& pointsfacet)
{
	p1 = pointsfacet(1,:);
	p2 = pointsfacet(2,:);
	p3 = pointsfacet(3,:);
	v1 = (p2-p1);
	v2 = (p3-p1);
	area = 0.5*sqrt((v1(2)*v2(3)-v1(3)*v2(2))^2 + (v1(3)*v2(1)-v1(1)*v2(3))^2 + (v1(1)*v2(2)-v1(2)*v2(1))^2);
	return area;
}

