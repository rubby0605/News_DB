#include <stdio.h>
#include <iostream>
#include <fstream>
int main(void)
{
using namespace std;
fstream file1("test_ip.txt");
 string stuff;

 while (getline(file1, stuff, '\n')) {
      cout << stuff << endl;
 }

 file1.close();
}
