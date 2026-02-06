#include<iostream>
#include<vector>
#include<algorithm>
#include<utility>
#include<atomic>

using namespace std;

struct MyArray { int z[50]; };

struct Mystr { int a, b};


int main(){
	std::atomic<MyArray> myArray;
	std::atomic<mystr> mystr;
	std::cout << std::boolalpha;
	std::cout << "std::atomic array is lock_free?";
	std::cout << std::atomic_is_lock_free<&myArray> <<std::endl;
	std::cout << "std::atomic mystr is lock_free?";
	std::cout << std::atomic_is_lock_free<&mystr> <<std::endl;
}
