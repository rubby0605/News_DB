#include<iostream>
#include<vector>
#include <algorithm>
using namespace std;
int main(){

std::vector<int> v={ 1,2,3,4,5,6,7,8,9};
std::string prefix="0";
for_each ( begin(v), end(v), [&prefix](int x) std::cout<<x<<std::endl;



}
