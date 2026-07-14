/* Copyright 2026 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#include <array>

#include "cp_test_helpers.h"
#include "cp_vector_funcs.h"

template<typename T>
void check_basic_vector_funcs() {
    constexpr uint64_t size = 9;
    std::array<T, size> a;
    std::array<T, size> b;
    std::array<T, size> out;
    for (uint64_t i = 0; i < size; ++i) {
        a[i] = static_cast<T>(i + 1);
        b[i] = static_cast<T>(2 * static_cast<int64_t>(i) - 3);
    }

    vec_mult_assign(out.data(), a.data(), static_cast<T>(1.5), size);
    for (uint64_t i = 0; i < size; ++i)
        EXPECT_NEAR(out[i], a[i] * static_cast<T>(1.5), static_cast<T>(1e-5));

    vec_mult_add(out.data(), b.data(), static_cast<T>(-0.25), size);
    for (uint64_t i = 0; i < size; ++i)
        EXPECT_NEAR(out[i], a[i] * static_cast<T>(1.5) - b[i] * static_cast<T>(0.25),
            static_cast<T>(1e-5));

    T expected_dot = static_cast<T>(0);
    for (uint64_t i = 0; i < size; ++i) expected_dot += a[i] * b[i];
    EXPECT_NEAR(dot_product(a.data(), b.data(), size), expected_dot, static_cast<T>(1e-4));

    vec_add_scaled(out.data(), a.data(), b.data(), static_cast<T>(0.5), size);
    for (uint64_t i = 0; i < size; ++i)
        EXPECT_NEAR(out[i], (a[i] + b[i]) * static_cast<T>(0.5), static_cast<T>(1e-5));

    std::array<T, size> prev;
    std::array<T, size> prev_m1;
    std::array<T, size> prev_prev_m1;
    std::array<T, size> gram_a;
    std::array<T, size> gram_b;
    for (uint64_t i = 0; i < size; ++i) {
        prev[i] = static_cast<T>(i + 1);
        prev_m1[i] = static_cast<T>(2 * i + 1);
        prev_prev_m1[i] = static_cast<T>(3 * static_cast<int64_t>(i) - 2);
        gram_a[i] = static_cast<T>(0.1 * (i + 1));
        gram_b[i] = static_cast<T>(0.05 * (i + 2));
    }
    auto idx = [](uint64_t i) { return size - i - 1; };
    vec_kernel_diag_step(out.data(), prev.data(), prev_m1.data(), prev_prev_m1.data(),
        gram_a.data(), gram_b.data(), idx, size);
    for (uint64_t i = 0; i < size; ++i) {
        const uint64_t gi = idx(i);
        const T expected = (prev[i] + prev_m1[i]) * gram_a[gi]
            - prev_prev_m1[i] * gram_b[gi];
        EXPECT_NEAR(out[i], expected, static_cast<T>(1e-4));
    }
}

TEST(VectorFuncsTest, BasicOperations) {
    check_basic_vector_funcs<float>();
    check_basic_vector_funcs<double>();
}

TEST(VectorFuncsTest, FourLaneOperations) {
    constexpr uint64_t count = 3;
    constexpr uint64_t lanes = 4;
    std::array<double, count * lanes> a;
    std::array<double, count * lanes> b;
    std::array<double, count * lanes> out;
    for (uint64_t i = 0; i < count * lanes; ++i) {
        a[i] = static_cast<double>(i + 1);
        b[i] = static_cast<double>(2 * static_cast<int64_t>(i) - 5);
    }

    vec4_add(out.data(), a.data(), b.data(), count);
    for (uint64_t i = 0; i < count * lanes; ++i) EXPECT_DOUBLE_EQ(out[i], a[i] + b[i]);

    vec4_add_inplace(out.data(), a.data(), count);
    for (uint64_t i = 0; i < count * lanes; ++i) EXPECT_DOUBLE_EQ(out[i], 2 * a[i] + b[i]);

    vec4_fmadd(out.data(), b.data(), 0.5, count);
    for (uint64_t i = 0; i < count * lanes; ++i)
        EXPECT_DOUBLE_EQ(out[i], 2 * a[i] + 1.5 * b[i]);

    vec4_scale(out.data(), a.data(), -2.0, count);
    for (uint64_t i = 0; i < count * lanes; ++i) EXPECT_DOUBLE_EQ(out[i], -2 * a[i]);
}

TEST(VectorFuncsTest, FourLaneBrackets) {
    constexpr uint64_t count = 3;
    constexpr uint64_t lanes = 4;
    std::array<double, count * lanes> v1;
    std::array<double, count * lanes> v2;
    for (uint64_t i = 0; i < count * lanes; ++i) {
        v1[i] = static_cast<double>(i + 1);
        v2[i] = static_cast<double>(i * i + 2);
    }

    const std::array<uint32_t, 2> ki = { 0, 1 };
    const std::array<uint32_t, 2> kj = { 1, 2 };
    const std::array<double, 2> kval = { 0.5, -2.0 };
    std::array<double, lanes> commutator;
    vec4_commutator_accum(commutator.data(), v1.data(), v2.data(),
        ki.data(), kj.data(), kval.data(), 0, 2);
    for (uint64_t lane = 0; lane < lanes; ++lane) {
        double expected = 0.0;
        for (uint64_t idx = 0; idx < ki.size(); ++idx) {
            expected += kval[idx] * (v1[ki[idx] * lanes + lane] * v2[kj[idx] * lanes + lane]
                - v1[kj[idx] * lanes + lane] * v2[ki[idx] * lanes + lane]);
        }
        EXPECT_DOUBLE_EQ(commutator[lane], expected);
    }

    const std::array<uint32_t, 2> ij_k = { 1, 2 };
    const std::array<double, 2> ij_c = { 2.0, -0.5 };
    std::array<double, count * lanes> scattered = {};
    vec4_bracket_scatter(scattered.data(), v1.data(), v2.data(), 0, 1,
        ij_k.data(), ij_c.data(), 0, 2);
    for (uint64_t lane = 0; lane < lanes; ++lane) {
        const double product = v1[lane] * v2[lanes + lane] - v1[lanes + lane] * v2[lane];
        EXPECT_DOUBLE_EQ(scattered[lanes + lane], 2.0 * product);
        EXPECT_DOUBLE_EQ(scattered[2 * lanes + lane], -0.5 * product);
    }

    std::array<double, count * lanes> dm_lf = {};
    std::array<double, count * lanes> dm_rf = {};
    vec4_bracket_grad(dm_lf.data(), dm_rf.data(), v2.data(), v1.data(), v2.data(),
        0, 1, ij_k.data(), ij_c.data(), 0, 2);
    for (uint64_t lane = 0; lane < lanes; ++lane) {
        const double sum = 2.0 * v2[lanes + lane] - 0.5 * v2[2 * lanes + lane];
        EXPECT_DOUBLE_EQ(dm_lf[lane], sum * v2[lanes + lane]);
        EXPECT_DOUBLE_EQ(dm_lf[lanes + lane], -sum * v2[lane]);
        EXPECT_DOUBLE_EQ(dm_rf[lanes + lane], sum * v1[lane]);
        EXPECT_DOUBLE_EQ(dm_rf[lane], -sum * v1[lanes + lane]);
    }
}

TEST(SparseMatrixTest, BasicTest1) {
    // Note the diagonal of 1s is assumed in both the matrix and the inverse
    SparseIntMatrix mat(4);
    mat.insert_entry(2, 0, 2);
    mat.insert_entry(3, 2, 3);

    SparseIntMatrix true_inv(4);
    true_inv.insert_entry(2, 0, -2);
    true_inv.insert_entry(3, 0, 6);
    true_inv.insert_entry(3, 2, -3);

    SparseIntMatrix inv;
    mat.inverse(inv);

    EXPECT_TRUE(true_inv == inv);
}

TEST(SparseMatrixTest, BasicTest2) {
    // Note the diagonal of 1s is assumed in both the matrix and the inverse
    SparseIntMatrix mat(5);
    mat.insert_entry(1, 0, 3);
    mat.insert_entry(2, 1, 1);
    mat.insert_entry(4, 0, 5);
    mat.insert_entry(4, 3, -2);

    SparseIntMatrix true_inv(5);
    true_inv.insert_entry(1, 0, -3);
    true_inv.insert_entry(2, 0, 3);
    true_inv.insert_entry(2, 1, -1);
    true_inv.insert_entry(4, 0, -5);
    true_inv.insert_entry(4, 3, 2);

    SparseIntMatrix inv;
    mat.inverse(inv);

    EXPECT_TRUE(true_inv == inv);
}

TEST(PathTest, ConstructorTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);
    Path<float> path2(std::span<float>(data), dimension, length);
    Path<float> path3(path2);

    EXPECT_TRUE(path == path2);
    EXPECT_TRUE(path == path3);
}

TEST(PathTest, SqBracketOperatorTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);
    Point<float> pt = path[3];
    EXPECT_EQ(static_cast<const float*>(data.data() + 3 * dimension), pt.data());
}

TEST(PathTest, FirstLastTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);

    Point<float> first = path.begin();
    Point<float> last = path.end();
    --last;

    for (uint64_t j = 0; j < dimension; ++j) {
        EXPECT_EQ(data[j], first[j]);
        EXPECT_EQ(data[(length - 1) * dimension + j], last[j]);
    }
}

#ifdef _DEBUG
TEST(PathTest, OutOfBoundsTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);

    try {
        path[length];
    }
    catch (const std::out_of_range& e) {
        EXPECT_EQ("Argument out of bounds in Path::operator[]", e.what());
    }
    catch (...) {
        FAIL();
    }
}
#endif

TEST(PointTest, ConstructorTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);

    Point<float> pt1(&path, 0);
    Point<float> pt2(&path, length - 1);
    Point<float> pt3(pt2);

    EXPECT_TRUE(pt1 != pt2);
    EXPECT_TRUE(pt2 == pt3);
}

TEST(PointTest, SqBracketOperatorTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);
    Point<float> pt(&path, 0);

    for (uint64_t i = 0; i < dimension; ++i)
        EXPECT_EQ(data[i], pt[i]);
}

TEST(PointTest, IncrementTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);
    Point<float> pt1(&path, 0);
    Point<float> pt2(&path, 0);

    for (uint64_t i = 0; i < length; ++i) {
        for (uint64_t j = 0; j < dimension; ++j) {
            EXPECT_EQ(data[i * dimension + j], pt1[j]);
            EXPECT_EQ(data[i * dimension + j], pt2[j]);
        }
        ++pt1;
        pt2++;
    }
}

TEST(PointTest, DecrementTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);
    Point<float> pt1 = --path.end();
    Point<float> pt2 = --path.end();

    for (int64_t i = length - 1; i >= 0; --i) {
        for (uint64_t j = 0; j < dimension; ++j) {
            EXPECT_EQ(data[i * dimension + j], pt1[j]);
            EXPECT_EQ(data[i * dimension + j], pt2[j]);
        }
        --pt1;
        pt2--;
    }
}

TEST(PointTest, AssignmentTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);
    Point<float> pt1 = path.begin();
    Point<float> pt2 = pt1;

    for (uint64_t i = 0; i < dimension; ++i) {
        EXPECT_EQ(data[i], pt1[i]);
        EXPECT_EQ(data[i], pt2[i]);
    }
}

TEST(PointTest, AdvanceTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);
    Point<float> pt(&path, 0);

    for (uint64_t i = 0; i < length; ++i) {
        for (uint64_t j = 0; j < dimension; ++j) {
            EXPECT_EQ(data[i * dimension + j], pt[j]);
        }
        pt.advance(1);
    }
}

TEST(PointTest, TimeAugTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length, true);

    int index = 0;

    for (Point<float> pt = path.begin(); pt != path.end(); ++pt) {
        for (int i = 0; i < dimension; i++) {
            float val = data[index * dimension + i];
            EXPECT_EQ(val, pt[i]);
        }
        EXPECT_TRUE(abs(static_cast<float>(index) / (length - 1) - pt[dimension]) < SINGLE_EPSILON);
        index++;
    }
}

TEST(PointTest, LeadLagTest) {
    uint64_t dimension = 2, length = 5;
    std::vector<float> data = {2, 6, 7, 1, 7, 0, 1, 7, 6, 3};
    std::vector<float> true_ = { 2, 6, 2, 6, 2, 6, 7, 1, 7, 1, 7, 1, 7, 1, 7, 0, 7, 0, 7, 0, 7, 0, 1, 7, 1, 7, 1, 7, 1, 7, 6, 3, 6, 3, 6, 3};

    Path<float> path(data.data(), dimension, length, false, true);

    int index = 0;
    bool parity = false;

    for (Point<float> pt = path.begin(); pt != path.end(); ++pt) {
        for (int i = 0; i < path.dimension(); ++i) {
            double val = pt[i];
            EXPECT_EQ(static_cast<double>(true_[index]), val);
            ++index;
        }
    }
}

TEST(PointTest, TimeAugLeadLagTest) {
    uint64_t dimension = 2, length = 5;
    std::vector<float> data = { 2, 6, 7, 1, 7, 0, 1, 7, 6, 3 };
    std::vector<double> true_ = { 2., 6., 2., 6., 0.,
        2., 6., 7., 1., 1. / 8,
        7., 1., 7., 1., 2. / 8,
        7., 1., 7., 0., 3. / 8,
        7., 0., 7., 0., 4. / 8,
        7., 0., 1., 7., 5. / 8,
        1., 7., 1., 7., 6. / 8,
        1., 7., 6., 3., 7. / 8,
        6., 3., 6., 3., 1. };

    Path<float> path(data.data(), dimension, length, true, true);

    int index = 0;
    bool parity = false;

    for (Point<float> pt = path.begin(); pt != path.end(); ++pt) {
        for (int i = 0; i < path.dimension(); ++i) {
            double val = pt[i];
            EXPECT_TRUE(abs(static_cast<double>(true_[index]) - pt[i]) < DOUBLE_EPSILON);
            ++index;
        }
    }
}

TEST(PointTest, ReverseTimeAugTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length, true);

    uint64_t index = length - 1;

    for (Point<float> pt = --path.end(); pt != --path.begin(); --pt) {
        for (int i = 0; i < dimension; i++) {
            float val = data[index * dimension + i];
            EXPECT_EQ(val, pt[i]);
        }
        EXPECT_TRUE(abs(static_cast<float>(index) / (length - 1) - pt[dimension]) < SINGLE_EPSILON);
        --index;
    }
}

#ifdef _DEBUG
TEST(PointTest, OutOfBoundsTest) {
    uint64_t dimension = 5, length = 10;
    std::vector<float> data = int_test_data(dimension, length);

    Path<float> path(data.data(), dimension, length);
    Point<float> pt = path.end();

    try { pt[0]; FAIL(); }
    catch (const std::out_of_range& e) { EXPECT_EQ("Point is out of bounds for given path in Point::operator[]", e.what()); }
    catch (...) { FAIL(); }

    pt = path.begin();
    try { pt[5]; FAIL(); }
    catch (const std::out_of_range& e) { EXPECT_EQ("Argument out of bounds in Point::operator[]", e.what()); }
    catch (...) { FAIL(); }

    Path<float> path2(path, true, false);
    pt = path2.begin();
    try { pt[5]; }
    catch (...) { FAIL(); }

    try { pt[6]; FAIL(); }
    catch (const std::out_of_range& e) { EXPECT_EQ("Argument out of bounds in Point::operator[]", e.what()); }
    catch (...) { FAIL(); }

    Path<float> path3(path, false, true);
    pt = path3.begin();
    try { pt[9]; }
    catch (...) { FAIL(); }

    try { pt[10]; FAIL(); }
    catch (const std::out_of_range& e) { EXPECT_EQ(e.what(), "Argument out of bounds in Point::operator[]"); }
    catch (...) { FAIL(); }

    Path<float> path4(path, true, true);
    pt = path4.begin();
    try { pt[10]; }
    catch (...) { FAIL(); }

    try { pt[11]; FAIL(); }
    catch (const std::out_of_range& e) { EXPECT_EQ("Argument out of bounds in Point::operator[]", e.what()); }
    catch (...) { FAIL(); }
}
#endif

TEST(transformPathForwardTest, TimeAugTest) {
    auto f = transform_path_d;
    uint64_t dimension = 2, length = 3;
    std::vector<double> input = {0., 0., 1., 2., 3., 4.};
    std::vector<double> true_ = {0., 0., 0., 1., 2., 0.5, 3., 4., 1.};
    check_result(f, input, true_, (uint64_t)1, (uint64_t)2, (uint64_t)3, true, false, 1., 1);
}

TEST(transformPathForwardTest, TimeAugCustomEndTime) {
    auto f = transform_path_d;
    uint64_t dimension = 2, length = 3;
    std::vector<double> input = {0., 0., 1., 2., 3., 4.};
    std::vector<double> true_ = {0., 0., 0., 1., 2., 1., 3., 4., 2.};
    check_result(f, input, true_, (uint64_t)1, (uint64_t)2, (uint64_t)3, true, false, 2., 1);
}

TEST(transformPathForwardTest, LeadLagTest) {
    auto f = transform_path_d;
    uint64_t dimension = 2, length = 3;
    std::vector<double> input = {1., 2., 3., 4., 5., 6.};
    std::vector<double> true_ = {1.,2.,1.,2., 1.,2.,3.,4., 3.,4.,3.,4., 3.,4.,5.,6., 5.,6.,5.,6.};
    check_result(f, input, true_, (uint64_t)1, (uint64_t)2, (uint64_t)3, false, true, 1., 1);
}

TEST(transformPathForwardTest, TimeAugLeadLagTest) {
    auto f = transform_path_d;
    uint64_t dimension = 2, length = 3;
    std::vector<double> input = {1., 2., 3., 4., 5., 6.};
    // lead_lag first (dim 2->4, len 3->5), then time_aug (dim 4->5, len 5)
    std::vector<double> true_ = {1.,2.,1.,2.,0., 1.,2.,3.,4.,0.25,
        3.,4.,3.,4.,0.5, 3.,4.,5.,6.,0.75, 5.,6.,5.,6.,1.};
    check_result(f, input, true_, (uint64_t)1, (uint64_t)2, (uint64_t)3, true, true, 1., 1);
}

TEST(transformPathForwardTest, BatchTimeAugTest) {
    auto f = transform_path_d;
    uint64_t batch_size = 2, dimension = 2, length = 3;
    std::vector<double> input = {0., 0., 1., 2., 3., 4.,
                                  5., 6., 7., 8., 9., 10.};
    uint64_t out_size = batch_size * (dimension + 1) * length;
    std::vector<double> true_(out_size);
    f(input.data(), true_.data(), batch_size, dimension, length, true, false, 1., 1);
    check_result(f, input, true_, batch_size, dimension, length, true, false, 1., -1);
}

TEST(transformPathBackprop, TimeAugTest) {
    auto f = transform_path_backprop_d;
    uint64_t dimension = 2, length = 3;
    std::vector<double> derivs((dimension + 1) * length, 1.);
    std::vector<double> true_ = { 1., 1., 1., 1., 1., 1. };
    check_result(f, derivs, true_, (uint64_t)1, dimension, length, true, false, 1., 1);
}

TEST(transformPathBackprop, LeadLagTest) {
    auto f = transform_path_backprop_d;
    uint64_t dimension = 2, length = 3;
    std::vector<double> derivs(2 * dimension * (2 * length - 1));
    for (int i = 0; i < derivs.size(); ++i)
        derivs[i] = i;
    std::vector<double> true_ = { 6., 9., 36., 40., 48., 51. };
    check_result(f, derivs, true_, (uint64_t)1, dimension, length, false, true, 1., 1);
}

TEST(transformPathBackprop, LeadLagTest2) {
    auto f = transform_path_backprop_d;
    uint64_t dimension = 5, length = 100;
    std::vector<double> derivs(2 * dimension * (2 * length - 1));
    for (uint64_t i = 0; i < derivs.size(); ++i)
        derivs[i] = 1.;
    std::vector<double> true_(dimension * length);
    for (uint64_t i = 0; i < dimension; ++i)
        true_[i] = 3.;
    for (uint64_t i = dimension; i < true_.size() - dimension; ++i)
        true_[i] = 4.;
    for (uint64_t i = true_.size() - dimension; i < true_.size(); ++i)
        true_[i] = 3.;
    check_result(f, derivs, true_, (uint64_t)1, dimension, length, false, true, 1., 1);
}

TEST(transformPathBackprop, TimeAugLeadLagTest) {
    auto f = transform_path_backprop_d;
    uint64_t dimension = 2, length = 3;
    std::vector<double> derivs((2 * dimension + 1) * (2 * length - 1), 1.);
    std::vector<double> true_ = { 3., 3., 4., 4., 3., 3. };
    check_result(f, derivs, true_, (uint64_t)1, dimension, length, true, true, 1., 1);
}

TEST(transformPathBackprop, BatchLeadLagTest) {
    auto f = transform_path_backprop_d;
    uint64_t batch_size = 2, dimension = 2, length = 3;
    std::vector<double> derivs(batch_size * 2 * dimension * (2 * length - 1));
    for (uint64_t i = 0; i < derivs.size(); ++i)
        derivs[i] = static_cast<double>(i);
    uint64_t out_size = batch_size * dimension * length;
    std::vector<double> true_(out_size);
    f(derivs.data(), true_.data(), batch_size, dimension, length, false, true, 1., 1);
    check_result(f, derivs, true_, batch_size, dimension, length, false, true, 1., -1);
}
