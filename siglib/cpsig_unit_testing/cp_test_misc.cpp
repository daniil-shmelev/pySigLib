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

#include "cp_test_helpers.h"

namespace cpSigTests {
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

    TEST_CLASS(transformPathBackprop) {
    public:

        TEST_METHOD(TimeAugTest) {
            auto f = transform_path_backprop_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs((dimension + 1) * length, 1.);
            std::vector<double> true_ = { 1., 1., 1., 1., 1., 1. };
            check_result(f, derivs, true_, (uint64_t)1, dimension, length, true, false, 1., 1);
        }
        TEST_METHOD(LeadLagTest) {
            auto f = transform_path_backprop_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs(2 * dimension * (2 * length - 1));
            for (int i = 0; i < derivs.size(); ++i)
                derivs[i] = i;
            std::vector<double> true_ = { 6., 9., 36., 40., 48., 51. };
            check_result(f, derivs, true_, (uint64_t)1, dimension, length, false, true, 1., 1);
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
            check_result(f, derivs, true_, (uint64_t)1, dimension, length, false, true, 1., 1);
        }

        TEST_METHOD(TimeAugLeadLagTest) {
            auto f = transform_path_backprop_d;
            uint64_t dimension = 2, length = 3;
            std::vector<double> derivs((2 * dimension + 1) * (2 * length - 1), 1.);
            std::vector<double> true_ = { 3., 3., 4., 4., 3., 3. };
            check_result(f, derivs, true_, (uint64_t)1, dimension, length, true, true, 1., 1);
        }
    };
}