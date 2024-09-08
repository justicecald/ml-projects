#include <iostream>

using namespace std;

bool palindrome(int const& decimal) {
    int v = decimal;
    int rev, rem = 0;
    while (v > 0) {
        rem += v % 10;
        rev = rem; // The lowest place value
        rem *= 10;
        v /= 10;
    }
    return (decimal == rev);
};

int main () {
    cout << "Palindrome? " << palindrome(899998) << endl;
    return 0;
}