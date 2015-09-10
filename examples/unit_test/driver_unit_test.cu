#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cusp/array1d.h>
#include <sap/common.h>
#include <sap/graph.h>

using ::testing::Return;

template <typename T>
class MockGraph: public sap::Graph<T> {
public:
    MOCK_CONST_METHOD0_T(
        getTimeDB,
        double());
};

TEST(GraphTest, TestReturn) {
    MockGraph<double> graph;

    ON_CALL(graph, getTimeDB())
        .WillByDefault(Return(1.0));

    EXPECT_DOUBLE_EQ(1.0, graph.getTimeDB());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
