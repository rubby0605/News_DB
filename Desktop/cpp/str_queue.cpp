#include <iostream>
using namespace std;

template<class>class shiftedlist
{
	T * array;
	int offset, size;
	public:
	shifted_list(int sz) :offset(0), size(sz) {
	array = new T[size];
	}
	~ShiftedList()
	{
		delete [] array;
	}
	void shiftby(int n)
	{
	offset = (offset + n) % size;
	}
	T getat(int i)
	{
		return array[ConvertIndex(i)];
	}
	
	void setAt(T item, int i)
	{
		array[ConverIndex(i)] = item;
	}
	
	private:
	int ConvertIndex(int i)
	{
		int index = (i - offset) % size;
		while (index < 0)  index += size;
		return index;
	}
};

	

