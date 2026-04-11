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

#pragma once
#include "cppch.h"
#include "disk_cache.h"

struct Entry {
    uint64_t col;
    int val;
};

class SparseIntMatrix {
public:
    uint64_t n;
    uint64_t m;
    std::vector<std::vector<Entry>> rows;

    SparseIntMatrix()
        : n(0), m(0), rows(0) {
    }

    SparseIntMatrix(uint64_t n_)
        : n(n_), m(n_), rows(n_) {
    }

    SparseIntMatrix(uint64_t n_, uint64_t m_)
        : n(n_), m(m_), rows(n_) {
    }

    SparseIntMatrix(SparseIntMatrix&& other) noexcept {
        n = other.n;
        m = other.m;
        rows.swap(other.rows);
    }

    void resize(uint64_t n_, uint64_t m_) {
        n = n_;
        m = m_;
        rows.resize(n);
    }

    void swap(SparseIntMatrix& other) noexcept {
        using std::swap;
        swap(n, other.n);
        swap(m, other.m);
        swap(rows, other.rows);
    }

    void populate_diagonal() {
#ifdef _DEBUG
        if (n != m) {
            throw std::runtime_error("n != m in SparseIntMatrix.populate_diagonal");
        }
#endif

        for (uint64_t i = 0; i < n; ++i) {
            this->insert_entry(i, i, 1);
        }
    }

    void drop_diagonal() {
#ifdef _DEBUG
        if (n != m) {
            throw std::runtime_error("n != m in SparseIntMatrix.drop_diagonal");
        }
#endif
        for (uint64_t i = 0; i < n; ++i) {
            auto& row = rows[i];
            for (auto it = row.begin(); it != row.end(); ++it) {
                if (it->col == i) {
                    row.erase(it);
                    break;
                }
            }
        }
    }

    bool is_nonzero(uint64_t i, uint64_t j) const {
#ifdef _DEBUG
        if (i >= n || j >= m) {
            throw std::out_of_range("i,j out of range in SparseIntMatrix.is_nonzero");
        }
#endif

        for (const auto& e : rows[i]) {
            if (e.col == j)
                return e.val != 0;
        }
        return false;
    }

    int get(uint64_t i, uint64_t j) const {
#ifdef _DEBUG
        if (i >= n || j >= m) {
            throw std::out_of_range("i,j out of range in SparseIntMatrix.get");
        }
#endif

        for (const auto& e : rows[i]) {
            if (e.col == j)
                return e.val;
        }
        return 0;
    }


    void insert_entry(uint64_t i, uint64_t j, int v) {
#ifdef _DEBUG
        if (i >= n || j >= m) {
            throw std::out_of_range("i,j out of range in SparseIntMatrix.insert_entry");
        }
#endif
        rows[i].emplace_back(j, v);
    }

    void drop_row(uint64_t i) {
#ifdef _DEBUG
        if (i >= n) {
            throw std::out_of_range("i out of range in SparseIntMatrix.drop_row");
        }
#endif
        rows.erase(rows.begin() + i);
        --n;
    }

    void inverse(SparseIntMatrix& out) const {
        // This assumes matrix is lower triangular with ones on the diagonal
        // The output will omit the diagonal.
#ifdef _DEBUG
        if (n != m) {
            throw std::runtime_error("n != m in SparseIntMatrix.inverse");
        }
#endif

        out.resize(n, n);

        for (uint64_t i = 0; i < n; ++i) {
            out.insert_entry(i, i, 1);
        }

        for (uint64_t i = 0; i < n; ++i) {
            std::unordered_map<uint64_t, int> row_i;

            for (const auto& e : rows[i]) {
                uint64_t k = e.col;
                int Lik = e.val;
                if (k >= i) continue;

                for (const auto& ek : out.rows[k]) {
                    uint64_t j = ek.col;
                    row_i[j] -= Lik * ek.val;
                }

                row_i[k] -= Lik;
            }

            out.rows[i].clear();
            for (const auto& [j, v] : row_i) {
                if (v != 0) {
                    out.rows[i].emplace_back(j, v);
                }
            }
        }
    }

    void transpose(SparseIntMatrix& out) const {
        out.resize(m, n);
        for (auto& row : out.rows) row.clear();

        for (uint64_t i = 0; i < n; ++i) {
            for (const auto& e : rows[i]) {
                out.rows[e.col].emplace_back(i, e.val);
            }
        }
    }

    template<std::floating_point T>
    void mul_vec_inplace_lower(T* arr) const {
        // This assumes matrix is lower triangular with ones on the diagonal
        for (uint64_t i_ = 0; i_ < n; ++i_) {
            uint64_t i = n - i_ - 1;
            for (const auto& e : rows[i]) {
                uint64_t j = e.col;
                arr[i] += static_cast<T>(e.val) * arr[j];
            }
        }
    }

    template<std::floating_point T>
    void mul_vec_inplace_upper(T* arr) const {
        // This assumes matrix is upper triangular with ones on the diagonal
        for (uint64_t i = 0; i < n; ++i) {
            for (const auto& e : rows[i]) {
                uint64_t j = e.col;
                arr[i] += static_cast<T>(e.val) * arr[j];
            }
        }
    }

    bool operator==(const SparseIntMatrix& other) const {
        if (n != other.n || m != other.m) return false;

        for (uint64_t i = 0; i < n; ++i) {
            std::unordered_map<uint64_t, int> row;

            for (const auto& e : rows[i])
                row[e.col] += e.val;

            for (const auto& e : other.rows[i])
                row[e.col] -= e.val;

            for (const auto& [j, v] : row) {
                if (v)
                    return false;
            }
        }

        return true;
    }

    void add_to_entry(uint64_t i, uint64_t j, int v) {
#ifdef _DEBUG
        if (i >= n || j >= m) {
            throw std::out_of_range("i,j out of range in SparseIntMatrix.add_to_entry");
        }
#endif
        for (auto& e : rows[i]) {
            if (e.col == j) {
                e.val += v;
                return;
            }
        }
        rows[i].emplace_back(j, v);
    }

    void serialize(std::ostream& out) const {
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(&m), sizeof(m));

        for (uint64_t i = 0; i < n; ++i) {
            uint64_t nnz = rows[i].size();
            out.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));

            for (const auto& e : rows[i]) {
                out.write(reinterpret_cast<const char*>(&e.col), sizeof(e.col));
                out.write(reinterpret_cast<const char*>(&e.val), sizeof(e.val));
            }
        }
    }

    static void deserialize(std::istream& in, SparseIntMatrix& out) {

        in.read(reinterpret_cast<char*>(&out.n), sizeof(out.n));
        in.read(reinterpret_cast<char*>(&out.m), sizeof(out.m));
        if (!in) throw std::runtime_error("Tried to read an invalid cache file: sparse matrix header");
        if (out.n > MAX_CACHE_VECTOR_SIZE || out.m > MAX_CACHE_VECTOR_SIZE)
            throw std::runtime_error("Tried to read an invalid cache file: sparse matrix dimension exceeds limit");

        check_stream_has_bytes(in, out.n * sizeof(uint64_t), "sparse matrix row headers");

        out.rows.clear();
        out.rows.resize(out.n);

        for (uint64_t i = 0; i < out.n; ++i) {
            uint64_t nnz;
            in.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
            if (!in) throw std::runtime_error("Tried to read an invalid cache file: sparse matrix nnz");
            if (nnz > out.m)
                throw std::runtime_error("Tried to read an invalid cache file: sparse matrix row nnz exceeds column count");

            if (nnz > 0)
                check_stream_has_bytes(in, nnz * (sizeof(uint64_t) + sizeof(int)), "sparse matrix row body");

            out.rows[i].resize(nnz);
            for (uint64_t k = 0; k < nnz; ++k) {
                in.read(reinterpret_cast<char*>(&out.rows[i][k].col), sizeof(uint64_t));
                in.read(reinterpret_cast<char*>(&out.rows[i][k].val), sizeof(int));
            }
            if (!in) throw std::runtime_error("Tried to read an invalid cache file: sparse matrix entry read");
        }
    }
};
