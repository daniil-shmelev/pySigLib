/* Copyright 2025 Daniil Shmelev
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

#include "CppUnitTest.h"
#include "cpsig.h"
#include "cp_tensor_poly.h"
#include "cp_path.h"
#include "cp_signature.h"
#include "cp_sig_kernel.h"
#include "sparse.h"
#include "words.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <span>
#include <cmath>

#define SINGLE_EPSILON 1e-4
#define DOUBLE_EPSILON 1e-10
#define EPSILON (std::is_same_v<T, float> ? SINGLE_EPSILON : DOUBLE_EPSILON)

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

double dot_product(double* a, double* b, uint64_t n) {
    double res = 0;
    for (int i = 0; i < n; ++i) {
        res += *(a + i) * *(b + i);
    }
    return res;
}

void gram_(
    double* path1,
    double* path2,
    double* out,
    uint64_t batch_size,
    uint64_t dimension,
    uint64_t length1,
    uint64_t length2
) {
    double* out_ptr = out;

    uint64_t flat_path1_length = length1 * dimension;
    uint64_t flat_path2_length = length2 * dimension;

    double* path1_start = path1;
    double* path1_end = path1 + flat_path1_length;

    double* path2_start = path2;
    double* path2_end = path2 + flat_path2_length;

    for (uint64_t b = 0; b < batch_size; ++b) {

        for (double* path1_ptr = path1_start; path1_ptr < path1_end - dimension; path1_ptr += dimension) {
            for (double* path2_ptr = path2_start; path2_ptr < path2_end - dimension; path2_ptr += dimension) {
                *(out_ptr++) = dot_product(path1_ptr + dimension, path2_ptr + dimension, dimension)
                    - dot_product(path1_ptr + dimension, path2_ptr, dimension)
                    - dot_product(path1_ptr, path2_ptr + dimension, dimension)
                    + dot_product(path1_ptr, path2_ptr, dimension);
            }
        }

        path1_start += flat_path1_length;
        path1_end += flat_path1_length;
        path2_start += flat_path2_length;
        path2_end += flat_path2_length;
    }
}


std::vector<float> int_test_data(uint64_t dimension, uint64_t length) {
    std::vector<float> data;
    uint64_t data_size = dimension * length;
    data.reserve(data_size);

    for (int i = 0; i < data_size; i++) {
        data.push_back(static_cast<float>(i));
    }
    return data;
}

template<typename FN, std::floating_point T, typename... Args>
void check_result(FN f, std::vector<T>& path, std::vector<T>& true_, Args... args) {
    std::vector<T> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    f(path.data(), out.data(), args...);

    for (uint64_t i = 0; i < true_.size(); ++i)
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs( - 1. - out[true_.size()]) < EPSILON);
}

template<typename FN, std::floating_point T, typename... Args>
void check_result_2(FN f, std::vector<T>& path1, std::vector<T>& path2, std::vector<T>& true_, Args... args) {
    std::vector<T> out;
    out.resize(true_.size() + 1); //+1 at the end just to check we don't write more than expected
    out[true_.size()] = -1.;

    f(path1.data(), path2.data(), out.data(), args...);

    for (uint64_t i = 0; i < true_.size(); ++i)
        Assert::IsTrue(abs(true_[i] - out[i]) < EPSILON);

    Assert::IsTrue(abs(-1. - out[true_.size()]) < EPSILON);
}

void check_result_words(std::vector<word> a, std::vector<word> b) {
    Assert::AreEqual(a.size(), b.size());
    for (uint64_t i = 0; i < a.size(); ++i) {
        Assert::AreEqual(a[i].size(), b[i].size());
        for (uint64_t j = 0; j < a[i].size(); ++j) {
            Assert::AreEqual(a[i][j], b[i][j]);
        }
    }
}

namespace cpSigTests
{
    TEST_CLASS(SparseMatrixTest) {
public:
    TEST_METHOD(BasicTest1)
    {
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

        Assert::IsTrue(true_inv == inv);
    }

    TEST_METHOD(BasicTest2)
    {
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

        Assert::IsTrue(true_inv == inv);
    }
    };

    TEST_CLASS(PolyTest)
    {
    public:
        TEST_METHOD(SigLengthTest)
        {
            Assert::AreEqual((uint64_t)1, sig_length(0, 0));
            Assert::AreEqual((uint64_t)1, sig_length(0, 1));
            Assert::AreEqual((uint64_t)1, sig_length(1, 0));

            Assert::AreEqual((uint64_t)435848050, sig_length(9, 9));
            Assert::AreEqual((uint64_t)11111111111, sig_length(10, 10));
            Assert::AreEqual((uint64_t)313842837672, sig_length(11, 11));

            Assert::AreEqual((uint64_t)10265664160401, sig_length(400, 5));

            Assert::AreEqual((uint64_t)0, sig_length(100, 100)); // overflow
        }

        TEST_METHOD(LogSigLengthTest)
        {
            Assert::AreEqual((uint64_t)0, log_sig_length(0, 0));
            Assert::AreEqual((uint64_t)0, log_sig_length(0, 1));
            Assert::AreEqual((uint64_t)0, log_sig_length(1, 0));

            Assert::AreEqual((uint64_t)5, log_sig_length(2, 3));

            Assert::AreEqual((uint64_t)49212093, log_sig_length(9, 9));
            Assert::AreEqual((uint64_t)1125217654, log_sig_length(10, 10));
            Assert::AreEqual((uint64_t)26039187, log_sig_length(5, 12));

            Assert::AreEqual((uint64_t)0, log_sig_length(100, 100)); // overflow
        }

        TEST_METHOD(PolyMultTestLinear)
        {
            // Test signatures of linear 2d paths
            auto f = sig_combine_d;
            std::vector<double> poly = { 1., 1., 1., 1./2, 1./2, 1./2, 1./2 };
            std::vector<double> true_res = { 1., 2., 2., 2., 2., 2., 2. };

            check_result_2(f, poly, poly, true_res, 2, 2);
        }

        TEST_METHOD(PolyMultSigTest)
        {
            uint64_t dimension = 2, length = 4, degree = 5;
            auto f = sig_combine_d;
            std::vector<double> path1 = { 0., 0., 1., 0.5, 0.4, 2. };
            std::vector<double> path2 = { 0.4, 2., 6., 0.1, 2.3, 4.1 };
            std::vector<double> path = { 0., 0., 1., 0.5, 0.4, 2., 6., 0.1, 2.3, 4.1 };

            uint64_t poly_len_ = sig_length(dimension, degree);

            std::vector<double> poly1;
            poly1.resize(poly_len_);
            signature_d(path1.data(), poly1.data(), dimension, 3, degree);

            std::vector<double> poly2;
            poly2.resize(poly_len_);
            signature_d(path2.data(), poly2.data(), dimension, 3, degree);

            std::vector<double> true_sig;
            true_sig.resize(poly_len_);
            signature_d(path.data(), true_sig.data(), dimension, 5, degree);
            check_result_2(f, poly1, poly2, true_sig, dimension, degree);
        }

        TEST_METHOD(BatchPolyMultSigTest)
        {
            uint64_t batch_size = 3, dimension = 2, length = 4, degree = 2;
            auto f = batch_sig_combine_d;
            std::vector<double> path1 = { 0., 0., 0.25, 0.25, 0.5, 0.5,
                0., 0., 0.4, 0.4, 0.6, 0.6,
                0., 0., 1., 0.5, 4., 0. };
            std::vector<double> path2 = { 0.5, 0.5, 1., 1.,
                0.6, 0.6, 1., 1.,
                4., 0., 0., 1. };
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            uint64_t res_len_ = sig_length(dimension, degree) * batch_size;

            std::vector<double> poly1;
            poly1.resize(res_len_);
            batch_signature_d(path1.data(), poly1.data(), batch_size, dimension, 3, degree);

            std::vector<double> poly2;
            poly2.resize(res_len_);
            batch_signature_d(path2.data(), poly2.data(), batch_size, dimension, 2, degree);

            std::vector<double> true_sig;
            true_sig.resize(res_len_);
            batch_signature_d(path.data(), true_sig.data(), batch_size, dimension, 4, degree);
            check_result_2(f, poly1, poly2, true_sig, batch_size, dimension, degree, 1);
            check_result_2(f, poly1, poly2, true_sig, batch_size, dimension, degree, -1);
        }

        TEST_METHOD(BatchPolyMultStressTest)
        {
            uint64_t batch_size = 1000, dimension = 5, degree = 5;

            std::vector<double> poly;
            poly.resize(batch_size * sig_length(dimension, degree));
            std::fill(poly.data(), poly.data() + poly.size(), 1.);

            std::vector<double> out;
            out.resize(batch_size * sig_length(dimension, degree));

            int err = batch_sig_combine_d(poly.data(), poly.data(), out.data(), batch_size, dimension, degree, -1);
            Assert::IsFalse(err);
        }
    };

    TEST_CLASS(PathTest)
    {
    public:
        TEST_METHOD(ConstructorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);
            Path<float> path2(std::span<float>(data), dimension, length);
            Path<float> path3(path2);

            Assert::IsTrue(path == path2);
            Assert::IsTrue(path == path3);
        }
        TEST_METHOD(SqBracketOperatorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);
            Point<float> pt = path[3];
            Assert::AreEqual(static_cast<const float*>(data.data() + 3 * dimension), pt.data());
        }
        TEST_METHOD(FirstLastTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);
            
            Point<float> first = path.begin();
            Point<float> last = path.end();
            --last;

            for (uint64_t j = 0; j < dimension; ++j){
                Assert::AreEqual(data[j], first[j]);
                Assert::AreEqual(data[(length - 1) * dimension + j], last[j]);
            }
        }

#ifdef _DEBUG
        TEST_METHOD(OutOfBoundsTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);

            try {
                path[length];
            }
            catch(const std::out_of_range& e){
                Assert::AreEqual("Argument out of bounds in Path::operator[]", e.what());
            }
            catch (...) {
                Assert::Fail();
            }

        }
#endif
    };

    TEST_CLASS(PointTest) {
    public:
        TEST_METHOD(ConstructorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);

            Point<float> pt1(&path, 0);
            Point<float> pt2(&path, length - 1);
            Point<float> pt3(pt2);

            Assert::IsTrue(pt1 != pt2);
            Assert::IsTrue(pt2 == pt3);
        }

        TEST_METHOD(SqBracketOperatorTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);
            Point<float> pt(&path, 0);

            for (uint64_t i = 0; i < dimension; ++i)
                Assert::AreEqual(data[i], pt[i]);
        }

        TEST_METHOD(IncrementTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);
            Point<float> pt1(&path, 0);
            Point<float> pt2(&path, 0);

            for (uint64_t i = 0; i < length; ++i) {
                for (uint64_t j = 0; j < dimension; ++j) {
                    Assert::AreEqual(data[i * dimension + j], pt1[j]);
                    Assert::AreEqual(data[i * dimension + j], pt2[j]);
                }
                ++pt1;
                pt2++;
            }
        }

        TEST_METHOD(DecrementTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);
            Point<float> pt1 = --path.end();
            Point<float> pt2 = --path.end();

            for (int64_t i = length - 1; i >= 0; --i) {
                for (uint64_t j = 0; j < dimension; ++j) {
                    Assert::AreEqual(data[i * dimension + j], pt1[j]);
                    Assert::AreEqual(data[i * dimension + j], pt2[j]);
                }
                --pt1;
                pt2--;
            }
        }

        TEST_METHOD(AssignmentTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);
            Point<float> pt1 = path.begin();
            Point<float> pt2 = pt1;

            for (uint64_t i = 0; i < dimension; ++i) {
                Assert::AreEqual(data[i], pt1[i]);
                Assert::AreEqual(data[i], pt2[i]);
            }
        }

        TEST_METHOD(AdvanceTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);
            Point<float> pt(&path, 0);

            for (uint64_t i = 0; i < length; ++i) {
                for (uint64_t j = 0; j < dimension; ++j) {
                    Assert::AreEqual(data[i * dimension + j], pt[j]);
                }
                pt.advance(1);
            }
        }
        TEST_METHOD(TimeAugTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length, true);

            int index = 0;

            for (Point<float> pt = path.begin(); pt != path.end(); ++pt) {
                for (int i = 0; i < dimension; i++) {
                    float val = data[index * dimension + i];
                    Assert::AreEqual(val, pt[i]);
                }
                Assert::IsTrue(abs(static_cast<float>(index) / (length - 1) - pt[dimension]) < SINGLE_EPSILON);
                index++;
            }
        }
        TEST_METHOD(LeadLagTest)
        {
            uint64_t dimension = 2, length = 5;
            std::vector<float> data = {2, 6, 7, 1, 7, 0, 1, 7, 6, 3};
            std::vector<float> true_ = { 2, 6, 2, 6, 2, 6, 7, 1, 7, 1, 7, 1, 7, 1, 7, 0, 7, 0, 7, 0, 7, 0, 1, 7, 1, 7, 1, 7, 1, 7, 6, 3, 6, 3, 6, 3};

            Path<float> path(data.data(), dimension, length, false, true);

            int index = 0;
            bool parity = false;

            for (Point<float> pt = path.begin(); pt != path.end(); ++pt) {
                for (int i = 0; i < path.dimension(); ++i) {
                    double val = pt[i];
                    Assert::AreEqual(static_cast<double>(true_[index]), val);
                    ++index;
                }
            }
        }
        TEST_METHOD(TimeAugLeadLagTest)
        {
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
                    Assert::IsTrue(abs(static_cast<double>(true_[index]) - pt[i]) < DOUBLE_EPSILON);
                    ++index;
                }
            }
        }

        TEST_METHOD(ReverseTimeAugTest)
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length, true);

            uint64_t index = length - 1;

            for (Point<float> pt = --path.end(); pt != --path.begin(); --pt) {
                for (int i = 0; i < dimension; i++) {
                    float val = data[index * dimension + i];
                    Assert::AreEqual(val, pt[i]);
                }
                Assert::IsTrue(abs(static_cast<float>(index) / (length - 1) - pt[dimension]) < SINGLE_EPSILON);
                --index;
            }
        }

#ifdef _DEBUG
        TEST_METHOD(OutOfBoundsTest) 
        {
            uint64_t dimension = 5, length = 10;
            std::vector<float> data = int_test_data(dimension, length);

            Path<float> path(data.data(), dimension, length);
            Point<float> pt = path.end();

            try { pt[0]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual("Point is out of bounds for given path in Point::operator[]", e.what()); }
            catch (...) { Assert::Fail(); }

            pt = path.begin();
            try { pt[5]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual("Argument out of bounds in Point::operator[]", e.what()); }
            catch (...) { Assert::Fail(); }

            Path<float> path2(path, true, false);
            pt = path2.begin();
            try { pt[5]; }
            catch (...) { Assert::Fail(); }

            try { pt[6]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual("Argument out of bounds in Point::operator[]", e.what()); }
            catch (...) { Assert::Fail(); }

            Path<float> path3(path, false, true);
            pt = path3.begin();
            try { pt[9]; }
            catch (...) { Assert::Fail(); }

            try { pt[10]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual(e.what(), "Argument out of bounds in Point::operator[]"); }
            catch (...) { Assert::Fail(); }

            Path<float> path4(path, true, true);
            pt = path4.begin();
            try { pt[10]; }
            catch (...) { Assert::Fail(); }

            try { pt[11]; Assert::Fail(); }
            catch (const std::out_of_range& e) { Assert::AreEqual("Argument out of bounds in Point::operator[]", e.what()); }
            catch (...) { Assert::Fail(); }
        }
#endif
    };

    TEST_CLASS(sigCoefDoubleTest)
    {

        TEST_METHOD(Trivial) {
            auto f = sig_coef_d;

            std::vector<double> path = { 0., 0. };
            std::vector<double> true_ = { 0., 0. };
            std::vector<uint64_t> multi_indices = { 0, 1 };
            std::vector<uint64_t> degrees = { 1, 1 };

            check_result(f, path, true_, multi_indices.data(), degrees.size(), degrees.data(), 2, 1, false, false, 1., false);
        }
        TEST_METHOD(Linear) {
            auto f = sig_coef_d;

            std::vector<double> path = { 0., 1. };
            std::vector<double> true_ = { 1., 1. / 2, 1. / 6, 1. / 24, 1. / 120 };
            std::vector<uint64_t> multi_indices(15, 0);
            std::vector<uint64_t> degrees = { 1, 2, 3, 4, 5 };

            check_result(f, path, true_, multi_indices.data(), degrees.size(), degrees.data(), 1, 2, false, false, 1., false);
        }

        TEST_METHOD(ManualSigTest) {
            auto f = sig_coef_d;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 0., 1., 0., 1., -1., 0.5 };

            std::vector<uint64_t> multi_indices = {
                0,
                1,
                0, 0,
                0, 1,
                1, 0,
                1, 1
            };

            std::vector<uint64_t> degrees = { 1, 1, 2, 2, 2, 2 };

            check_result(f, path, true_sig, multi_indices.data(), degrees.size(), degrees.data(), dimension, length, false, false, 1., false);
        }

        TEST_METHOD(BatchManualSigTest) {
            auto f = batch_sig_coef_d;
            uint64_t batch_size = 2, dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1.,
            0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 0., 1., 0., 1., -1., 0.5,
            0., 1., 0., 1., -1., 0.5 };

            std::vector<uint64_t> multi_indices = {
                0,
                1,
                0, 0,
                0, 1,
                1, 0,
                1, 1
            };

            std::vector<uint64_t> degrees = { 1, 1, 2, 2, 2, 2 };

            check_result(f, path, true_sig, multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, false, false, 1., false, 1);
        }

        TEST_METHOD(ManualSigTestPrefixes) {
            auto f = sig_coef_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 0., 1., -1., 1., 0.5, -0.5 };

            std::vector<uint64_t> multi_indices = {
                0,
                1, 0,
                1, 1, 0
            };

            std::vector<uint64_t> degrees = { 1, 2, 3 };

            check_result(f, path, true_sig, multi_indices.data(), degrees.size(), degrees.data(), dimension, length, false, false, 1., true);
        }

        TEST_METHOD(BatchManualSigTestPrefixes) {
            auto f = batch_sig_coef_d;
            uint64_t batch_size = 2, dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1.,
            0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 0., 1., -1., 1., 0.5, -0.5,
            0., 1., -1., 1., 0.5, -0.5 };

            std::vector<uint64_t> multi_indices = {
                0,
                1, 0,
                1, 1, 0
            };

            std::vector<uint64_t> degrees = { 1, 2, 3 };

            check_result(f, path, true_sig, multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, false, false, 1., true, 1);
        }
    };

    TEST_CLASS(sigCoefBackpropTest) {
    public:

        TEST_METHOD(ManualTest1) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 2, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5 };
            std::vector<double> true_deriv = { -1./6, -1./6, 1./6, 1./6 };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0
            };

            std::vector<double> coefs = { 1., 0.25, 1./12 };
            std::vector<double> derivs = { 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), dimension, length, false, false, 1.);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 3, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0. };
            std::vector<double> true_deriv = { -1. / 6, -10./6, 2./3, -1. - 1./3, -0.5, 3.};

            std::vector<uint64_t> multi_indices = {
                0, 1, 0
            };

            std::vector<double> coefs = { 4., -1., -2./3 };
            std::vector<double> derivs = { 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), dimension, length, false, false, 1.);
        }
        TEST_METHOD(ManualTest3) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 0.5 + 1./3, 1./3, 2./3, 20./3, -1.-1./6, -1.-2./3, -1./3, -5-1./3 };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0
            };

            std::vector<double> coefs = {0., 1., -2.};
            std::vector<double> derivs = { 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), dimension, length, false, false, 1.);
        }

        TEST_METHOD(ManualTest4) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 19./24, 17./12, 2./3, 7.+1./3, -31./24, -25./12, -1./6, -20./3 };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0,
                1, 1, 0
            };

            std::vector<double> coefs = { 0., 1., -2., 1., 0.5, -0.5 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), dimension, length, false, false, 1.);
        }

        TEST_METHOD(ManualTestBatch) {
            auto f = batch_sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3, batch_size = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1., 0., 0., 2., 0., 1., 2., 0., 1. };
            std::vector<double> true_deriv = { 19. / 24, 17. / 12, 2. / 3, 7. + 1. / 3, -31. / 24, -25. / 12, -1. / 6, -20. / 3, 2.5, 3.5 + 1./3, -4., 2./3, -2.5, -3.5, 4., -1. };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0,
                1, 1, 0
            };

            std::vector<double> coefs = { 0., 1., -2., 1., 0.5, -0.5, 0., 2.5, -4-1./3, 1., 0.5, -1.5-1./3 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, false, false, 1., 1);
        }

        TEST_METHOD(ManualTest4Empty) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 19./24, 17./12, 2./3, 7.+1./3, -31./24, -25./12, -1./6, -20./3 };

            std::vector<uint64_t> multi_indices = {
                0, 1, 0,
                1, 1, 0
            };

            std::vector<double> coefs = { 1., 0., 1., -2., 1., 0.5, -0.5 };
            std::vector<double> derivs = { 1., 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 0, 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), dimension, length, false, false, 1.);
        }

        TEST_METHOD(ManualTestTimeAug) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 55./36, 29./18, -1., -4./9, -71./36, -17./18, 13./9, -2./9 };

            std::vector<uint64_t> multi_indices = {
                0, 2, 0,
                1, 2, 0
            };

            std::vector<double> coefs = { 0., 1. + 2./3, -4. - 2./9, 1., 1./3, -4./9 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), dimension, length, true, false, 1.);
        }

        TEST_METHOD(ManualTestTimeAugBatch) {
            auto f = batch_sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3, batch_size = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1.,
            0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { 55. / 36, 29. / 18, -1., -4. / 9, -71. / 36, -17. / 18, 13. / 9, -2. / 9,
            55. / 36, 29. / 18, -1., -4. / 9, -71. / 36, -17. / 18, 13. / 9, -2. / 9 };

            std::vector<uint64_t> multi_indices = {
                0, 2, 0,
                1, 2, 0
            };

            std::vector<double> coefs = { 0., 1. + 2. / 3, -4. - 2. / 9, 1., 1. / 3, -4. / 9,
            0., 1. + 2. / 3, -4. - 2. / 9, 1., 1. / 3, -4. / 9 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1.,
            0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), batch_size, dimension, length, true, false, 1., 1);
        }

        TEST_METHOD(ManualTestLeadLag) {
            auto f = sig_coef_backprop_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_deriv = { -13., -.25, -5.375, 1.5, 47.375, -2.75, -29., 1.5 };

            std::vector<uint64_t> multi_indices = {
                0, 2, 0,
                1, 2, 3
            };

            std::vector<double> coefs = { 0., -13., 61., 1., 1.5, 1.125 };
            std::vector<double> derivs = { 0., 0., 1., 0., 0., 1. };

            std::vector<uint64_t> degrees = { 3, 3 };

            check_result(f, path, true_deriv, coefs.data(), derivs.data(), multi_indices.data(), degrees.size(), degrees.data(), dimension, length, false, true, 1.);
        }
    };

    TEST_CLASS(signatureDoubleTest)
    {
    public:
        TEST_METHOD(TrivialCases) {
            auto f = signature_d;
            std::vector<double> path;
            std::vector<double> true_sig;
            Assert::AreEqual(2, f(path.data(), true_sig.data(), 0, 0, 0, false, false, 1., true));

            true_sig.push_back(1.);
            check_result(f, path, true_sig, 1, 0, 0, false, false, 1., true);

            path.push_back(0.);
            check_result(f, path, true_sig, 1, 1, 0, false, false, 1., true);

            true_sig.push_back(0.);
            check_result(f, path, true_sig, 1, 0, 1, false, false, 1., true);
            check_result(f, path, true_sig, 1, 1, 1, false, false, 1., true);

            path.push_back(1.);
            true_sig[1] = 1.;
            check_result(f, path, true_sig, 1, 2, 1, false, false, 1., true);
        }
        TEST_METHOD(LinearPathTest) {
            auto f = signature_d;
            uint64_t dimension = 2, length = 3, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
            std::vector<double> true_sig;
            true_sig.resize(level_4_start);
            true_sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { true_sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { true_sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { true_sig[i] = 1 / 6.; }
            check_result(f, path, true_sig, dimension, length, degree, false, false, 1., true);
        }

        TEST_METHOD(LinearPathTest2) {
            auto f = signature_d;
            uint64_t dimension = 2, length = 4, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> path = { 0.,0., 0.25, 0.25, 0.75, 0.75, 1.,1. };
            std::vector<double> true_sig;
            true_sig.resize(level_4_start);
            true_sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { true_sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { true_sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { true_sig[i] = 1 / 6.; }
            check_result(f, path, true_sig, dimension, length, degree, false, false, 1., true);
        }

        TEST_METHOD(ManualSigTest) {
            auto f = signature_d;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 1., 0.5, 4., 0., 0., 1. };
            std::vector<double> true_sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            check_result(f, path, true_sig, dimension, length, degree, false, false, 1., true);
        }
        TEST_METHOD(ManualSigTest2) {
            auto f = signature_f;
            uint64_t dimension = 3, length = 4, degree = 3;
            std::vector<float> path = {
                 9.f, 5.f, 8.f, 5.f, 3.f, 0.f, 0.f, 2.f, 6.f, 4.f, 0.f, 2.f
            };

            std::vector<float> true_sig = {
                 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                 5.f, 0.5f, 12.5f, 9.f, 25.f,
                 21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                 33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f
            };
            check_result(f, path, true_sig, dimension, length, degree, false, false, 1.f, true);
        }

        TEST_METHOD(BatchSigTest) {
            auto f = batch_signature_d;
            uint64_t dimension = 2, length = 4, degree = 2;
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> true_sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 0., 1., 0., 1., -1., 0.5 };

            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, 1);
            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, -1);
        }

        TEST_METHOD(BatchSigTestDegree1) {
            auto f = batch_signature_d;
            uint64_t dimension = 2, length = 4, degree = 1;
            std::vector<double> path = { 0., 0., 0.25, 0.25, 0.5, 0.5, 1., 1.,
                0., 0., 0.4, 0.4, 0.6, 0.6, 1., 1.,
                0., 0., 1., 0.5, 4., 0., 0., 1. };

            std::vector<double> true_sig = { 1., 1., 1.,
                1., 1., 1.,
                1., 0., 1. };

            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, 1);
            check_result(f, path, true_sig, 3, dimension, length, degree, false, false, 1., true, -1);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = signature_f;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<float> path = { 0.f, 5.f, 2.f, 4.f, 9.f };
            std::vector<float> true_sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                                64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f/3.f };
            float end_time = length - 1.f;
            check_result(f, path, true_sig, dimension, length, degree, true, false, end_time, true);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = signature_f;
            uint64_t dimension = 1, length = 5, degree = 3;
            std::vector<float> path = { 0.f, 5.f, 2.f, 4.f, 9.f };
            std::vector<float> true_sig = { 1.f, 9.f, 9.f, 40.5f, 9.f, 72.f, 40.5f, 121.5f, 6.5f, 68.f, -8.5f, 290.f, 98.f, 275.f, 121.5f };
            check_result(f, path, true_sig, dimension, length, degree, false, true, 1.f, true);
        }

        TEST_METHOD(BigLeadLagTest) {
            auto f = batch_signature_d;
            uint64_t dimension = 2, length = 10, degree = 2, batch = 1;
            std::vector<double> path;
            path.resize(batch * length * dimension);
            std::vector<double> out;
            out.resize(batch * sig_length(dimension * 2, degree));
            f(path.data(), out.data(), batch, dimension, length, degree, false, true, 1., true, 1);
        }
    };

    TEST_CLASS(sigBackpropTest) {
    public:
        TEST_METHOD(LinearPathTest) {
            auto f = sig_backprop_d;
            uint64_t dimension = 2, length = 2, degree = 2;
            std::vector<double> path = { 0., 0., 1.,1. };
            std::vector<double> deriv = { 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -3., -3., 3., 3. };
            std::vector<double> sig = {1., 1., 1., 1./2, 1./2, 1./2, 1./2};
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, false, false, 1.);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_backprop_d;
            uint64_t dimension = 2, length = 3, degree = 2;
            std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { -7.5, -10., -0.5, 0.25, 8., 9.75 };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, false, false, 1.);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_backprop_d;
            uint64_t dimension = 2, length = 3, degree = 3;
            std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { -19.625, -23.625, -1.25, 0.625, 20.875, 23. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, false, false, 1.);
        }

        TEST_METHOD(ManualTestAsBatch) {
            auto f = batch_sig_backprop_d;
            uint64_t dimension = 2, length = 3, degree = 2;
            std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { -7.5, -10., -0.5, 0.25, 8., 9.75 };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result(f, path, true_, deriv.data(), sig.data(), 1, dimension, length, degree, false, false, 1., 1);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            auto f = batch_sig_backprop_d;
            uint64_t dimension = 2, length = 3, degree = 3;
            std::vector<double> path = { 0., 0., 1.,2., 0.5, 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { -19.625, -23.625, -1.25, 0.625, 20.875, 23. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result(f, path, true_, deriv.data(), sig.data(), 1, dimension, length, degree, false, false, 1., 1);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = batch_sig_backprop_d;
            uint64_t dimension = 2, length = 3, degree = 3, batch_size = 3;
            std::vector<double> path = { 0., 0., 1., 2., 0.5, 1., 0., 0., 3., 2., 5., 2., 0., 0., -1., 2., 0.5, -1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12., 13., -14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -19.625, -23.625, -1.25, 0.625, 20.875, 23., -162.5, -103.5, -81.0, 245.5, 243.5, -142.0, -0.625, -0.625, 0., 0., 0.625, 0.625 };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1./48, 1./24, 1./24, 1./12, 1./24, 1./12, 1./12, 1./6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5./6, 3., 9., 2., 13., 2., 6., 1. + 1./3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1./48, -1./24, -1./24,  1./12, -1./24, 1./48, 1./48, -1./6 };
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, false, 1., 1);
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, false, 1., -1);
        }

        TEST_METHOD(TimeAugTest) {
            auto f = sig_backprop_d;
            uint64_t dimension = 1, length = 3, degree = 3;
            std::vector<double> path = { 0., 2., 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { -54., -4.5, 58.5 };
            std::vector<double> sig = { 1., 1., 2., 0.5, 2.5, -0.5, 2., 1./6, 1.5 + 1./3, -1-1./6, 2 + 1./6, 1./3, 2./3, -0.5-1./3, 1 + 1./3 };
            double end_time = length - 1.;
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, true, false, end_time);
        }

        TEST_METHOD(BatchTimeAugTest) {
            auto f = batch_sig_backprop_d;
            uint64_t dimension = 1, length = 3, degree = 3, batch_size = 2;
            std::vector<double> path = { 0., 2., 1., 0., 3., 6. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -54., -4.5, 58.5, -41., 0., 41. };
            std::vector<double> sig = { 1., 1., 2., 0.5, 2.5, -0.5, 2., 1. / 6, 1.5 + 1. / 3, -1 - 1. / 6, 2 + 1. / 6, 1. / 3, 2. / 3, -0.5 - 1. / 3, 1 + 1. / 3,
            1., 6., 2., 18., 6., 6., 2., 36., 12., 12., 4., 12., 4., 4., 4./3};
            double end_time = length - 1.;
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, false, end_time, 1);
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, false, end_time, -1);
        }

        TEST_METHOD(LeadLagTest) {
            auto f = sig_backprop_d;
            uint64_t dimension = 1, length = 3, degree = 3;
            std::vector<double> path = { 0., 2., 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { -76., 5.5, 70.5 };
            std::vector<double> sig = { 1., 1., 1., .5, -2., 3., .5, 1./6, -2., 2., 1., .5, -4., 3.5, 1./6 };
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, false, true, 1.);
        }

        TEST_METHOD(BatchLeadLagTest) {
            auto f = batch_sig_backprop_d;
            uint64_t dimension = 1, length = 3, degree = 3, batch_size = 2;
            std::vector<double> path = { 0., 2., 1., 0., 3., 6. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -76., 5.5, 70.5, -170., 0., 170. };
            std::vector<double> sig = { 1., 1., 1., .5, -2., 3., .5, 1. / 6, -2., 2., 1., .5, -4., 3.5, 1. / 6,
            1., 6., 6., 18., 9., 27., 18., 36., 13.5, 27., 13.5, 67.5, 27., 67.5, 36. };
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, true, 1., 1);
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, false, true, 1., -1);
        }

        TEST_METHOD(TimeAugLeadLagTest) {
            auto f = sig_backprop_d;
            uint64_t dimension = 1, length = 3, degree = 2;
            std::vector<double> path = { 0., 2., 1. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. };
            std::vector<double> true_ = { -98., -6., 104. };
            std::vector<double> sig = { 1., 1., 1., 4., .5, -2., 4.5, 3., .5, 5.5, -.5, -1.5, 8. };
            double end_time = length * 2. - 2.;
            check_result(f, path, true_, deriv.data(), sig.data(), dimension, length, degree, true, true, end_time);
        }

        TEST_METHOD(BatchTimeAugLeadLagTest) {
            auto f = batch_sig_backprop_d;
            uint64_t dimension = 1, length = 3, degree = 2, batch_size = 2;
            std::vector<double> path = { 0., 2., 1., 0., 3., 6. };
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { -98., -6., 104., -34., 0., 34. };
            std::vector<double> sig = { 1., 1., 1., 4., .5, -2., 4.5, 3., .5, 5.5, -.5, -1.5, 8.,
            1., 6., 6., 4., 18., 9., 9., 27., 18., 15., 15., 9., 8. };
            double end_time = length * 2. - 2.;
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, true, end_time, 1);
            check_result(f, path, true_, deriv.data(), sig.data(), batch_size, dimension, length, degree, true, true, end_time, -1);
        }
    };

    TEST_CLASS(sigCombineBackpropTest) {
    public:
        TEST_METHOD(ManualTest) {
            auto f = sig_combine_backprop_d;
            uint64_t dimension = 2, degree = 2;
            uint64_t result_length = 7;
            std::vector<double> sig1 = { 1., 1., 1., .5, .5, .5, .5 };
            std::vector<double> sig2 = { 1., 0., 1., 0., 1., -1., .5 };
            std::vector<double> derivs = {1., 1., 2., 3., 4., 5., 6.};
            std::vector<double> true_ = { 1., 5., 8., 3., 4., 5., 6., 1., 9., 12., 3., 4., 5., 6. };

            auto func = [&](double* sig_combined_derivs, double* out, double* sig1, double* sig2, uint64_t dimension, uint64_t degree) {
                f(sig_combined_derivs, out, out + result_length, sig1, sig2, dimension, degree);
                };


            check_result(func, derivs, true_, sig1.data(), sig2.data(), dimension, degree);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = batch_sig_combine_backprop_d;
            uint64_t dimension = 2, degree = 2, batch_size = 2;
            uint64_t result_length = 7 * batch_size;
            std::vector<double> sig1 = { 1., 1., 1., .5, .5, .5, .5, 
                1., 0., 1., 0., 1., -1., .5 };
            std::vector<double> sig2 = { 1., 0., 1., 0., 1., -1., .5, 
                1., 1., 1., .5, .5, .5, .5 };
            std::vector<double> derivs = { 1., 1., 2., 3., 4., 5., 6., 
                1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 1., 5., 8., 3., 4., 5., 6., 
                1., 8., 13., 3., 4., 5., 6., 
                1., 9., 12., 3., 4., 5., 6., 
                1., 6., 8., 3., 4., 5., 6. };

            auto func = [&](double* sig_combined_derivs, double* out, double* sig1, double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) {
                f(sig_combined_derivs, out, out + result_length, sig1, sig2, batch_size, dimension, degree, n_jobs);
                };


            check_result(func, derivs, true_, sig1.data(), sig2.data(), batch_size, dimension, degree, 1);
            check_result(func, derivs, true_, sig1.data(), sig2.data(), batch_size, dimension, degree, -1);
        }
    };

    TEST_CLASS(lyndonWordsTest) {
    public:
        TEST_METHOD(AllLyndonWordsTest1) {
            uint64_t dimension = 2, degree = 3;
            std::vector<word> result = all_lyndon_words(dimension, degree);
            std::vector<word> true_ = {
                {0}, {1},
                {0,1},
                {0,0,1}, {0,1,1}
            };
            check_result_words(result, true_);
        }

        TEST_METHOD(AllLyndonWordsTest2) {
            uint64_t dimension = 5, degree = 2;
            std::vector<word> result = all_lyndon_words(dimension, degree);
            std::vector<word> true_ = {
                {0}, {1}, {2}, {3}, {4},
                {0,1}, {0,2}, {0,3}, {0,4},
                {1,2}, {1,3}, {1,4},
                {2,3}, {2,4}, {3,4}
            };
            check_result_words(result, true_);
        }

        TEST_METHOD(AllLyndonWordsTest3) {
            uint64_t dimension = 3, degree = 4;
            std::vector<word> result = all_lyndon_words(dimension, degree);
            std::vector<word> true_ = { 
                { 0 }, {1}, {2},
                {0, 1}, {0, 2}, {1, 2}, 
                {0, 0, 1}, {0, 0, 2}, {0, 1, 1}, {0, 1, 2}, {0, 2, 1}, {0, 2, 2}, {1, 1, 2}, {1, 2, 2},
                {0, 0, 0, 1}, {0, 0, 0, 2}, {0, 0, 1, 1}, {0, 0, 1, 2}, {0, 0, 2, 1}, {0, 0, 2, 2},
                {0, 1, 0, 2}, {0, 1, 1, 1}, {0, 1, 1, 2}, {0, 1, 2, 1}, {0, 1, 2, 2}, {0, 2, 1, 1},
                {0, 2, 1, 2}, {0, 2, 2, 1}, {0, 2, 2, 2}, {1, 1, 1, 2}, {1, 1, 2, 2}, {1, 2, 2, 2}
            };
            check_result_words(result, true_);
        }
    };

    TEST_CLASS(lyndonMatrixTest) {
    public:
        TEST_METHOD(lyndonMatrixTest1) {
            uint64_t dimension = 2, degree = 2;
            std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
            std::vector<uint64_t> lyndon_idx = all_lyndon_idx(dimension, degree);
            SparseIntMatrix out;
            lyndon_proj_matrix(out, lyndon_words, lyndon_idx, dimension, degree);

            SparseIntMatrix true_(out.n);

            Assert::IsTrue(true_ == out);
        }
        TEST_METHOD(lyndonMatrixTest2) {
            uint64_t dimension = 3, degree = 4;
            std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
            std::vector<uint64_t> lyndon_idx = all_lyndon_idx(dimension, degree);
            SparseIntMatrix out;
            lyndon_proj_matrix(out, lyndon_words, lyndon_idx, dimension, degree);

            SparseIntMatrix true_(out.n);
            true_.insert_entry(10, 9, -1);
            true_.insert_entry(18, 17, -1);
            true_.insert_entry(20, 18, -1);
            true_.insert_entry(23, 22, -2);
            true_.insert_entry(25, 22, 1);
            true_.insert_entry(25, 23, -1);
            true_.insert_entry(26, 24, -2);
            true_.insert_entry(27, 24, 1);
            true_.insert_entry(27, 26, -1);

            Assert::IsTrue(true_ == out);
        }

        TEST_METHOD(lyndonMatrixTest3) {
            uint64_t dimension = 2, degree = 5;
            std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
            std::vector<uint64_t> lyndon_idx = all_lyndon_idx(dimension, degree);
            SparseIntMatrix out;
            lyndon_proj_matrix(out, lyndon_words, lyndon_idx, dimension, degree);

            SparseIntMatrix true_(out.n);
            true_.insert_entry(10, 9, -2);
            true_.insert_entry(12, 11, -3);

            Assert::IsTrue(true_ == out);
        }
    };

    TEST_CLASS(logSignatureExpandedTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            check_result(f, sig, true_, dimension, degree, false, false, 0);
        }

        TEST_METHOD(LinearPathTest2) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            check_result(f, sig, true_, dimension, degree, false, false, 0);
        }

        TEST_METHOD(ManualLogSigTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> true_ = { 0., 0., 1., 0., 1., -1., 0. };
            std::vector<double> sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            check_result(f, sig, true_, dimension, degree, false, false, 0);
        }

        TEST_METHOD(ManualLogSigTest2) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 3, degree = 3;
            std::vector<float> true_ = {
                 0.f, -5.f, -5.f, -6.f, 0.f, 12.f, -10.f, -12.f,
                 0.f, -6.f, 10.f, 6.f, 0.f, 0.f, -27.f,
                 11.f, 54.f, 5.f, 3.f + 2.f / 3.f, -22.f, 20.f + 2.f / 3.f, -18.f,
                -27.f, -10.f, -24.f - 1.f / 3.f, 5.f, 0.f, -9.f, 20.f + 2.f / 3.f,
                 18.f, -4.f - 2.f / 3.f, 11.f, -24.f - 1.f / 3.f, 36.f, 3.f + 2.f / 3.f, -9.f,
                 9.f + 1.f / 3.f, -18.f, -4.f - 2.f / 3.f, 0.f
            };

            std::vector<float> sig = {
                 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                 5.f, 0.5f, 12.5f, 9.f, 25.f,
                 21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                 33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f
            };
            check_result(f, sig, true_, dimension, degree, false, false, 0);
        }

        TEST_METHOD(BatchLogSigTest) {
            auto f = batch_sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2;

            std::vector<double> true_ = { 0., 1., 1., 0., 0., 0., 0.,
                0., 1., 1., 0., 0., 0., 0.,
                0., 0., 1., 0., 1., -1., 0.};

            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 1., 1., 0.5, 0.5, 0.5, 0.5,
                1., 0., 1., 0., 1., -1., 0.5 };

            check_result(f, sig, true_, 3, dimension, degree, false, false, 0, 1);
            check_result(f, sig, true_, 3, dimension, degree, false, false, 0, -1);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 1, degree = 3;
            std::vector<float> true_ = { 0.f, 9.f, 4.f, 0.f, -2.5f, 2.5f, 0.f, 0.f, -5.25f,
                                10.5f, 5.5f, -5.25f, -11.f, 5.5f, 0.f};
            std::vector<float> sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                                64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f / 3.f };
            check_result(f, sig, true_, dimension, degree, true, false, 0);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 1, degree = 3;
            std::vector<float> true_ = { 0., 9., 9., 0., -31.5, 31.5, 0., 0., 26.75, -53.5, 11.75, 26.75, -23.5, 11.75, 0. };
            std::vector<float> sig = { 1., 9., 9., 40.5, 9., 72., 40.5, 121.5, 6.5, 68., -8.5, 290., 98., 275., 121.5 };
            check_result(f, sig, true_, dimension, degree, false, true, 0);
        }

        TEST_METHOD(BigLeadLagTest) {
            auto f = batch_sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2, batch = 1;
            std::vector<double> out;
            out.resize(batch * sig_length(dimension * 2, degree));
            std::vector<double> sig;
            sig.resize(batch * sig_length(dimension * 2, degree));
            f(sig.data(), out.data(), batch, dimension, degree, false, true, 0, 1);
        }
    };

    TEST_CLASS(logSignatureExpandedBackpropTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 1., -1., -1.,  1.,  1.,  1.,  1. };
            std::vector<double> sig = {1., 1., 1., 0.5, 0.5, 0.5, 0.5};
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 0);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 1., -5., -6.25, 3., 4., 5., 6. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 0);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { 1., 6.5, 7.6875, -10, -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 0);
        }

        TEST_METHOD(ManualTestAsBatch) {
            auto f = batch_sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 1., -5., -6.25, 3., 4., 5., 6. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 0, 1);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            auto f = batch_sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> true_ = { 1., 6.5, 7.6875, -10, -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 0, 1);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = batch_sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3, batch_size = 3;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 1., 1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12., 13., -14., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 1., 6.5, 7.6875, -10., -11.25, -12.5, -13.75, 7., 8., 9., 10., 11., 12., 13., 14., 1., 66., 30.25, -35., 15.5, -46., 14.5, 7., -8., 9., -10., 11., -12., 13., -14., 1., 1.625, 1.625, 1.5, 1.5, 1.5, 1.5, 1., 1., 1., 1., 1., 1., 1., 1. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 0, 1);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 0, -1);
        }

        TEST_METHOD(ManualDim1Test) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 1, degree = 8;
            std::vector<double> deriv = { 1., 1., 2., 3., 4., 5., 6., 7., 8. };
            std::vector<double> true_ = { 1., -1., 8., 9., 1., -8., -9., -1., 8. };
            std::vector<double> sig = { 1., 1., 2., 3., 4., 5., 6., 7., 8. };
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 0);
        }
    };

    TEST_CLASS(logSignatureLyndonWordsTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, dimension, degree, false, false, 1);
        }

        TEST_METHOD(LinearPathCacheTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            clear_cache(true); // Clear disk
            prepare_log_sig(dimension, degree, 1, true);
            clear_cache(false); // Remove from memory but keep on disk
            check_result(f, sig, true_, dimension, degree, false, false, 1);
        }

        TEST_METHOD(LinearPathTest2) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, dimension, degree, false, false, 1);
        }

        TEST_METHOD(ManualLogSigTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> true_ = { 0., 1., 1. };
            std::vector<double> sig = { 1., 0., 1., 0., 1., -1., 0.5 };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, dimension, degree, false, false, 1);
        }

        TEST_METHOD(ManualLogSigTest2) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 3, degree = 3;
            std::vector<float> true_ = {
                -5.f, -5.f, -6.f, 12.f, -10.f, -6.f, -27.f,
                 11.f, 5.f, 3.f + 2.f / 3.f, 20.f + 2.f / 3.f, -18.f, -9.f, -4.f - 2.f / 3.f
            };

            std::vector<float> sig = {
                 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                 5.f, 0.5f, 12.5f, 9.f, 25.f,
                 21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                 33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f
            };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, dimension, degree, false, false, 1);
        }

        TEST_METHOD(BatchLogSigTest) {
            auto f = batch_sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> true_ = { 1., 1., 0.,
                1., 1., 0.,
                0., 1., 1. };

            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5,
               1., 1., 1., 0.5, 0.5, 0.5, 0.5,
               1., 0., 1., 0., 1., -1., 0.5 };

            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, 3, dimension, degree, false, false, 1, 1);
            check_result(f, sig, true_, 3, dimension, degree, false, false, 1, -1);
        }

        TEST_METHOD(ManualTimeAugTest) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 1, degree = 3;
            std::vector<float> true_ = { 9.f, 4.f, -2.5f, -5.25f, 5.5f };
            std::vector<float> sig = { 1.f, 9.f, 4.f, 40.5f, 15.5f, 20.5f, 8.f, 121.5f, 37.5f,
                                64.5f, 24.5f, 60.f, 13.f, 34.5f, 10.f + 2.f / 3.f };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, dimension, degree, true, false, 1);
        }

        TEST_METHOD(ManualLeadLagTest) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 1, degree = 3;
            std::vector<float> true_ = { 9.f, 9.f, -31.5f, 26.75f, 11.75f };
            std::vector<float> sig = { 1.f, 9.f, 9.f, 40.5f, 9.f, 72.f, 40.5f, 121.5f, 6.5f, 68.f, -8.5f, 290.f, 98.f, 275.f, 121.5f };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, dimension, degree, false, true, 1);
        }

        TEST_METHOD(BigLeadLagTest) {
            auto f = batch_sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 2, batch = 1;
            std::vector<double> out;
            out.resize(batch * sig_length(dimension * 2, degree));
            std::vector<double> sig;
            sig.resize(batch * sig_length(dimension * 2, degree));
            prepare_log_sig(dimension, degree, 1);
            f(sig.data(), out.data(), batch, dimension, degree, false, true, 1, 1);
        }
    };

    TEST_CLASS(logSignatureLyndonWordsBackpropTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 1. };
            std::vector<double> true_ = { 0., .5, .5, 0., 1., 0., 0. };
            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 1);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 1);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 1);
        }

        TEST_METHOD(ManualTestAsBatch) {
            auto f = batch_sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 1, 1);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            auto f = batch_sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 1, 1);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = batch_sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3, batch_size = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 1., -2., 3., -4., 5., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0., 0., -21., 8., 4., 8., 0., -12.5, 0., -4., 0., 5., 0., 0., 0., 0., 0., 1.375, 0.5625, 0.5, 1.25, 0., -0.25, 0., 1., 0., 1., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
            prepare_log_sig(dimension, degree, 1);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 1, 1);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 1, -1);
        }
    };

    TEST_CLASS(logSignatureLyndonBasisTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, dimension, degree, false, false, 2);
        }

        TEST_METHOD(LinearPathCacheTest) {
            auto f = sig_to_log_sig_d;
            uint64_t dimension = 2, degree = 3;
            uint64_t level_3_start = sig_length(dimension, 2);
            uint64_t level_4_start = sig_length(dimension, 3);
            std::vector<double> true_ = { 1., 1., 0., 0., 0. };
            std::vector<double> sig;
            sig.resize(level_4_start);
            sig[0] = 1.;
            for (uint64_t i = 1; i < dimension + 1; ++i) { sig[i] = 1.; }
            for (uint64_t i = dimension + 1; i < level_3_start; ++i) { sig[i] = 1 / 2.; }
            for (uint64_t i = level_3_start; i < level_4_start; ++i) { sig[i] = 1 / 6.; }
            clear_cache(true); // Clear disk
            prepare_log_sig(dimension, degree, 2, true);
            clear_cache(false); // Clear memory
            check_result(f, sig, true_, dimension, degree, false, false, 2);
        }

        TEST_METHOD(ManualLogSigTest2) {
            auto f = sig_to_log_sig_f;
            uint64_t dimension = 3, degree = 3;
            std::vector<float> true_ = { -5.f, -5.f, -6.f, 12.f, -10.f, -6.f, -27.f,
            11.f, 5.f, 3.f + 2.f / 3.f, 24.f + 1.f / 3.f, -18.f, -9.f, -4.f - 2.f / 3.f };
            std::vector<float> sig = { 1.f, -5.f, -5.f, -6.f, 12.5f, 24.5f,
                                                5.f, 0.5f, 12.5f, 9.f, 25.f,
                                               21.f, 18.f, -20.5f - 1.f / 3.f, -77.5f - 1.f / 3.f, 11.f,
                                               33.f + 1.f / 6.f, -45.5f - 1.f / 3.f, -42.f - 1.f / 3.f, -47.f, 5.f + 2.f / 3.f,
                                              -18.f, -17.5f - 1.f / 3.f, -30.5f - 1.f / 3.f, 11.f + 2.f / 3.f, 14.f + 1.f / 6.f,
                                              -20.5f - 1.f / 3.f, -19.f, -14.f - 1.f / 3.f, -7.f, -16.f - 2.f / 3.f,
                                              -39.f, -110.f - 1.f / 3.f, 6.f, -1.f / 3.f, -49.f,
                                              -20.f - 2.f / 3.f, -78.f, -52.f - 2.f / 3.f, -36.f };
            prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, dimension, degree, false, false, 2);
        }
    };

    TEST_CLASS(logSignatureLyndonBasisBackpropTest) {
    public:

        TEST_METHOD(LinearPathTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 1., 1. };
            std::vector<double> true_ = { 0., .5, .5, 0., 1., 0., 0. };
            std::vector<double> sig = { 1., 1., 1., 0.5, 0.5, 0.5, 0.5 };
            prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 2);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 2);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), dimension, degree, false, false, 2);
        }

        TEST_METHOD(ManualTestAsBatch) {
            auto f = batch_sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 2;
            std::vector<double> deriv = { 1., 2., 3. };
            std::vector<double> true_ = { 0., -0.5, 1.25, 0., 3., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5 };
            prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 2, 1);
        }

        TEST_METHOD(ManualTest2AsBatch) {
            auto f = batch_sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 6. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -0.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6 };
            prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), 1, dimension, degree, false, false, 2, 1);
        }

        TEST_METHOD(ManualBatchTest) {
            auto f = batch_sig_to_log_sig_backprop_d;
            uint64_t dimension = 2, degree = 3, batch_size = 3;
            std::vector<double> deriv = { 1., 2., 3., 4., 5., 1., -2., 3., -4., 5., 1., 1., 1., 1., 1. };
            std::vector<double> true_ = { 0., 0.75, 2.375, -2., -.5, 0., -1.25, 0., 4., 0., 5., 0., 0., 0., 0., 0., -21., 8., 4., 8., 0., -12.5, 0., -4., 0., 5., 0., 0., 0., 0., 0., 1.375, 0.5625, 0.5, 1.25, 0., -0.25, 0., 1., 0., 1., 0., 0., 0., 0. };
            std::vector<double> sig = { 1., 0.5, 1., 0.125, 0.25, 0.25, 0.5, 1. / 48, 1. / 24, 1. / 24, 1. / 12, 1. / 24, 1. / 12, 1. / 12, 1. / 6, 1., 5., 2., 12.5, 3., 7., 2., 20. + 5. / 6, 3., 9., 2., 13., 2., 6., 1. + 1. / 3, 1., 0.5, -1., 0.125, -0.25, -0.25, 0.5, 1. / 48, -1. / 24, -1. / 24,  1. / 12, -1. / 24, 1. / 48, 1. / 48, -1. / 6 };
            prepare_log_sig(dimension, degree, 2);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 2, 1);
            check_result(f, sig, true_, deriv.data(), batch_size, dimension, degree, false, false, 2, -1);
        }
    };

    TEST_CLASS(sigKernelTest) {
    public:

        TEST_METHOD(Trivial) {
            auto f = sig_kernel_d;
            uint64_t dimension = 1, length = 1;
            std::vector<double> path = { 0. };
            std::vector<double> true_sig = { 1. };
            std::vector<double> gram = {};
            check_result(f, gram, true_sig, dimension, length, length, 0, 0, false);
        }

        TEST_METHOD(TrivialBatch) {
            auto f = batch_sig_kernel_d;
            uint64_t dimension = 1, length = 1, batch_size = 5;
            std::vector<double> path = { 0. };
            std::vector<double> true_sig = { 1., 1., 1., 1., 1. };
            std::vector<double> gram = {};
            check_result(f, gram, true_sig, batch_size, dimension, length, length, 0, 0, 1, false);
        }
        TEST_METHOD(LinearPathTest) {
            auto f = sig_kernel_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> path = { 0., 0., 0.5, 0.5, 1.,1. };
            std::vector<double> true_sig = { 4.256702149748847 };
            std::vector<double> gram((length - 1) * (length - 1));
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            check_result(f, gram, true_sig, dimension, length, length, 2, 2, false);
        }

        TEST_METHOD(ManualTest) {
            auto f = sig_kernel_d;
            uint64_t dimension = 3, length = 4;
            std::vector<double> path = { .9, .5, .8, .5, .3, .0, .0, .2, .6, .4, .0, .2 };
            std::vector<double> true_sig = { 2.1529809076880486 };
            std::vector<double> gram((length - 1) * (length - 1));
            gram_(path.data(), path.data(), gram.data(), 1, dimension, length, length);
            check_result(f, gram, true_sig, dimension, length, length, 2, 2, false);
        }

        TEST_METHOD(NonSquare1) {
            auto f = sig_kernel_d;
            uint64_t dimension = 1, length1 = 3, length2 = 2;
            std::vector<double> path1 = { 0., 1., 2. };
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> true_sig = { 11. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, dimension, length1, length2, 0, 0, false);
        }

        TEST_METHOD(NonSquare2) {
            auto f = sig_kernel_d;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path2 = { 0., 1., 2. };
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> true_sig = { 11. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, dimension, length1, length2, 0, 0, false);
        }

        TEST_METHOD(FullGrid) {
            auto f = sig_kernel_d;
            uint64_t dimension = 1, length1 = 3, length2 = 2;
            std::vector<double> path1 = { 0., 1., 2. };
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> true_sig = { 1., 1.,
                1., 4.,
                1., 11. };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_sig, dimension, length1, length2, 0, 0, true);
        }
    };

    TEST_CLASS(sigKernelBackpropTest) {
    public:
        TEST_METHOD(ManualTest1) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> path2 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 4.5 + 1./6, 4.5 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, k_grid.data(), dimension, length1, length2, 0, 0);
        }

        TEST_METHOD(ManualTest1Extended) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length1 = 34, length2 = 35;
            std::vector<double> path1(length1, 0.);
            path1[length1 - 1] = 2.;
            std::vector<double> path2(length2, 0.);
            path2[length2 - 2] = 1.;
            path2[length2 - 1] = 2.;
            double deriv = 1.;
            std::vector<double> true_((length1 - 1) * (length2 - 1), 11.); //{ 4.5 + 1. / 6, 4.5 };

            for (uint64_t i = 1; i < length1 - 1; ++i) {
                true_[(length2 - 1) * i - 2] = 7. + 1. / 9;
                true_[(length2 - 1) * i - 1] = 2. + 1. / 3;
            }
            for (uint64_t i = (length1 - 2) * (length2 - 1); i < (length1 - 1) * (length2 - 1) - 2; ++i) {
                true_[i] = 5. + 4. / 9;
            }

            true_[(length1 - 1) * (length2 - 1) - 2] = 4.5 + 1. / 6;
            true_[(length1 - 1) * (length2 - 1) - 1] = 4.5;
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid(length1 * length2, 1.);// = { 1., 1., 1., 1., 4., 11. };
            k_grid[length1 * length2 - 2] = 4.;
            k_grid[length1 * length2 - 1] = 11.;
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, k_grid.data(), dimension, length1, length2, 0, 0);
        }

        TEST_METHOD(ManualTest1Rev) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length2 = 2, length1 = 3;
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> path1 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 4.5 + 1. / 6, 4.5 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 4., 1., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, k_grid.data(), dimension, length1, length2, 0, 0);
        }

        TEST_METHOD(ManualTest2) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length1 = 3, length2 = 3;
            std::vector<double> path1 = { 0., 2., 3. };
            std::vector<double> path2 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 761./72, 7.125, 133./24, 12.5 + 1. / 6 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 7., 25. - 1. / 6 };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, k_grid.data(), dimension, length1, length2, 0, 0);
        }

        TEST_METHOD(ManualTest2Rev) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length2 = 3, length1 = 3;
            std::vector<double> path2 = { 0., 2., 3. };
            std::vector<double> path1 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 761. / 72, 133. / 24, 7.125, 12.5 + 1. / 6 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 7., 1., 11., 25. - 1. / 6 };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, k_grid.data(), dimension, length1, length2, 0, 0);
        }

        TEST_METHOD(ManualTest3) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2. };
            std::vector<double> path2 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 5.1602194279800226, 5.1185673607720270 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = { 
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.5625,
                2.27734375,
                3.1857910156249996,
                4.3402760823567705,
                1.0,
                2.27734375,
                4.25830078125,
                7.2303009033203125,
                11.584854549831814
            };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, k_grid.data(), dimension, length1, length2, 1, 1);
        }

        TEST_METHOD(ManualTest3Rev) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 1, length2 = 2, length1 = 3;
            std::vector<double> path2 = { 0., 2. };
            std::vector<double> path1 = { 0., 1., 2. };
            double deriv = 1.;
            std::vector<double> true_ = { 5.1602194279800226, 5.1185673607720270 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = {
                1.0,
                1.0,
                1.0,
                1.0,
                1.5625,
                2.27734375,
                1.0,
                2.27734375,
                4.25830078125,
                1.0,
                3.1857910156249996,
                7.2303009033203125,
                1.0,
                4.3402760823567705,
                11.584854549831814
            };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, k_grid.data(), dimension, length1, length2, 1, 1);
        }

        TEST_METHOD(ManualTest4) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 2, length1 = 3, length2 = 3;
            std::vector<double> path1 = { 0., 1., 2., 4., 5., 5. };
            std::vector<double> path2 = { 0., 2., 1., 3., 2., 1. };
            double deriv = 1.;
            std::vector<double> true_ = { 1631. / 72, -437. / 96, 817. / 32, 1049. / 24 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = {
                1.0,
                1.0,
                1.0,
                1.0,
                12.25,
                4.75,
                1.0,
                57.75,
                87.729 + 1./6000
            };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, k_grid.data(), dimension, length1, length2, 0, 0);
        }

        TEST_METHOD(ManualTest4Rev) {
            auto f = sig_kernel_backprop_d;
            uint64_t dimension = 2, length2 = 3, length1 = 3;
            std::vector<double> path2 = { 0., 1., 2., 4., 5., 5. };
            std::vector<double> path1 = { 0., 2., 1., 3., 2., 1. };
            double deriv = 1.;
            std::vector<double> true_ = { 1631. / 72, 817. / 32 , -437. / 96, 1049. / 24 };
            std::vector<double> gram((length1 - 1) * (length2 - 1));
            std::vector<double> k_grid = {
                1.0,
                1.0,
                1.0,
                1.0,
                12.25,
                57.75,
                1.0,
                4.75,
                87.729 + 1. / 6000
            };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            check_result(f, gram, true_, deriv, k_grid.data(), dimension, length1, length2, 0, 0);
        }

        TEST_METHOD(BatchManualTest1) {
            auto f = batch_sig_kernel_backprop_d;
            uint64_t batch_size = 2, dimension = 1, length1 = 2, length2 = 3;
            std::vector<double> path1 = { 0., 2., 0., 2. };
            std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
            std::vector<double> derivs = { 1., 1. };
            std::vector<double> true_ = { 4.5 + 1. / 6, 4.5, 4.5 + 1. / 6, 4.5 };
            std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 1., 1., 1., 4., 11. };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            gram_(path1.data(), path2.data(), gram.data() + 2, 1, dimension, length1, length2);
            check_result(f, gram, true_, derivs.data(), k_grid.data(), batch_size, dimension, length1, length2, 0, 0, 1);
        }

        TEST_METHOD(BatchManualTest2) {
            auto f = batch_sig_kernel_backprop_d;
            uint64_t batch_size = 2, dimension = 1, length1 = 3, length2 = 3;
            std::vector<double> path1 = { 0., 2., 3., 0., 2., 3. };
            std::vector<double> path2 = { 0., 1., 2., 0., 1., 2. };
            std::vector<double> derivs = { 1., 1. };
            std::vector<double> true_ = { 761. / 72, 7.125, 133. / 24, 12.5 + 1. / 6, 761. / 72, 7.125, 133. / 24, 12.5 + 1. / 6 };
            std::vector<double> gram((length1 - 1) * (length2 - 1) * batch_size);
            std::vector<double> k_grid = { 1., 1., 1., 1., 4., 11., 1., 7., 25. - 1. / 6, 1., 1., 1., 1., 4., 11., 1., 7., 25. - 1. / 6 };
            gram_(path1.data(), path2.data(), gram.data(), 1, dimension, length1, length2);
            gram_(path1.data(), path2.data(), gram.data() + 4, 1, dimension, length1, length2);
            check_result(f, gram, true_, derivs.data(), k_grid.data(), batch_size, dimension, length1, length2, 0, 0, 1);
        }
    };

    TEST_CLASS(transformPathBackprop) {
    public:

        TEST_METHOD(TimeAugTest) {
            auto f = transform_path_backprop_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs((dimension + 1) * length, 1.);
            std::vector<double> true_ = { 1., 1., 1., 1., 1., 1. };
            check_result(f, derivs, true_, dimension, length, true, false, 1.);
        }
        TEST_METHOD(LeadLagTest) {
            auto f = transform_path_backprop_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs(2 * dimension * (2 * length - 1));
            for (int i = 0; i < derivs.size(); ++i)
                derivs[i] = i;
            std::vector<double> true_ = { 6., 9., 36., 40., 48., 51. };
            check_result(f, derivs, true_, dimension, length, false, true, 1.);
        }

        TEST_METHOD(LeadLagTest2) {
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
            check_result(f, derivs, true_, dimension, length, false, true, 1.);
        }

        TEST_METHOD(TimeAugLeadLagTest) {
            auto f = transform_path_backprop_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs((2 * dimension + 1) * (2 * length - 1), 1.);
            std::vector<double> true_ = { 3., 3., 4., 4., 3., 3. };
            check_result(f, derivs, true_, dimension, length, true, true, 1.);
        }
    };
}
//TODO: add tests for transform_path