#include <gtest/gtest.h>
#include <cusp/array1d.h>

typedef typename cusp::array1d<int, cusp::host_memory>    IntVectorH;

bool isEven(int n) {
    return (n & 1) == 0;
}

TEST(IsEvenTest, BlahBlah) {
    IntVectorH test_v(10, 4);
    EXPECT_TRUE(isEven(2));
    EXPECT_FALSE(isEven(3));

    for (int i = 0; i < test_v.size(); i++) {
        EXPECT_TRUE(isEven(test_v[i]));
    }
}
