// Copyright 2026. Licensed under the Apache License, Version 2.0.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef __CUDACC__
#define BSK_DEVICE __device__
#else
#define BSK_DEVICE
#endif

inline uint64_t branched_poly_mul(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::invalid_argument("branched polynomial kernel dimensions overflow");
    return a * b;
}

inline uint64_t branched_poly_add(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<size_t>::max() - b)
        throw std::invalid_argument("branched polynomial kernel workspace overflows");
    return a + b;
}

struct BranchedPolyDims {
    uint64_t nx, ny, q, height, width, nodes, cells, state, scratch, output;
    BranchedPolyDims(uint64_t length1, uint64_t length2, uint64_t depth,
                    uint64_t order, bool grid, uint64_t batch, uint64_t dimension) {
        if (!length1 || !length2)
            throw std::invalid_argument("branched polynomial kernel paths must have length >= 1");
        if (!dimension)
            throw std::invalid_argument("branched polynomial kernel dimension must be positive");
        if (order > 64)
            throw std::invalid_argument("branched polynomial kernel order must be between 0 and 64");
        nx = length1 - 1;
        ny = length2 - 1;
        q = order + 1;
        height = branched_poly_mul(nx, q);
        width = branched_poly_mul(ny, q);
        cells = branched_poly_mul(nx, ny);
        nodes = branched_poly_mul(height, width);
        state = depth > 1 ? branched_poly_mul(depth, nodes) : 0;
        output = grid ? branched_poly_mul(length1, length2) : 1;
        scratch = branched_poly_add(branched_poly_mul(3, nodes), branched_poly_add(cells, width));
        branched_poly_mul(batch, branched_poly_mul(state, sizeof(double)));
        branched_poly_mul(batch, branched_poly_mul(output, sizeof(double)));
        branched_poly_mul(scratch, sizeof(double));
    }
};

// The first three regions hold nodal values and separable-integration scratch.
// CPU integration also uses a column-prefix buffer. A reverse call without
// saved state appends the depth stack after scratch; forward never accesses it.
template<typename T>
struct BranchedPolyWorkspace {
    T* values;
    T* next_values;
    T* integrated;
    T* cell_integrals;
    T* column_prefix;
    T* recomputed_state;

    BSK_DEVICE BranchedPolyWorkspace(T* data, const BranchedPolyDims& d)
        : values(data), next_values(data + d.nodes), integrated(data + 2*d.nodes),
          cell_integrals(data + 3*d.nodes), column_prefix(cell_integrals + d.cells),
          recomputed_state(data + d.scratch) {}
};

// Evaluate K_0,...,K_(depth-1), optionally retaining every depth for reverse.
// Ops supplies serial CPU loops or cooperative CUDA block operations.
template<typename T, class Ops>
BSK_DEVICE const T* branched_poly_depths(
    const T* gram, T* state, BranchedPolyWorkspace<T>& work,
    const BranchedPolyDims& d, uint64_t depth, const T* tables, Ops ops
) {
    const T* weights = tables + d.q*d.q;
    T* initial = state ? state : work.values;
    for (uint64_t k = ops.index(); k < d.nodes; k += ops.stride()) initial[k] = T(1);
    ops.sync();
    const T* previous = initial;
    for (uint64_t m = 1; m < depth; ++m) {
        T* current = state ? state + m*d.nodes
                           : (previous == work.values ? work.next_values : work.values);
        for (uint64_t k = ops.index(); k < d.nodes; k += ops.stride())
            current[k] = gram[(k/d.width/d.q)*d.ny + (k%d.width)/d.q]*previous[k];
        ops.sync();
        ops.integrate_x(current, work.integrated, d.nx, d.width, d.q, tables, weights, false);
        ops.integrate_y(work.integrated, current, d.height, d.ny, d.q, tables, weights, false);
        ops.exp(current, d.nodes);
        ops.sync();
        previous = current;
    }
    return previous;
}

// Whole-cell quadrature at the final depth. The returned sum is local to each
// worker; scalar output reduces it, whereas grid output uses a 2D prefix sum.
template<typename T, class Ops>
BSK_DEVICE T branched_poly_integrate_cells(
    const T* gram, const T* previous, T* cells,
    const BranchedPolyDims& d, const T* weights, Ops ops
) {
    T local = 0;
    for (uint64_t k = ops.index(); k < d.cells; k += ops.stride()) {
        const uint64_t i = k/d.ny, j = k%d.ny;
        T integral = 0;
        for (uint64_t r = 0; r < d.q; ++r)
            for (uint64_t s = 0; s < d.q; ++s)
                integral += weights[r]*weights[s]*previous[(i*d.q+r)*d.width+j*d.q+s];
        cells[k] = gram[k]*integral;
        local += cells[k];
    }
    ops.sync();
    return local;
}

template<typename T, class Ops>
BSK_DEVICE void branched_poly_forward_pair(
    const T* gram, T* out, T* state, BranchedPolyWorkspace<T>& work,
    const BranchedPolyDims& d, uint64_t depth, const T* tables, bool grid, Ops ops
) {
    const T* previous = branched_poly_depths(gram, state, work, d, depth, tables, ops);
    const T local = branched_poly_integrate_cells(
        gram, previous, work.cell_integrals, d, tables + d.q*d.q, ops);
    if (grid) {
        ops.prefix(work.cell_integrals, d.nx, d.ny, false);
        ops.exp_grid(work.cell_integrals, out, d.nx, d.ny);
    } else {
        const T value = exp(ops.sum(local));
        if (ops.index() == 0) *out = value;
    }
}

template<typename T, class Ops>
BSK_DEVICE void branched_poly_backprop_pair(
    const T* gram, T* grad, const T* derivs, const T* state,
    BranchedPolyWorkspace<T>& work, const BranchedPolyDims& d,
    uint64_t depth, const T* tables, bool grid, Ops ops
) {
    if (!state) {
        branched_poly_depths(gram, work.recomputed_state, work, d, depth, tables, ops);
        state = work.recomputed_state;
    }
    const T* weights = tables + d.q*d.q;
    const T* previous = state + (depth-1)*d.nodes;
    T* cells = work.cell_integrals;
    T* adjoint = work.values; // Forward scratch is free once the depth stack exists.
    const T local = branched_poly_integrate_cells(gram, previous, cells, d, weights, ops);
    if (grid) {
        ops.prefix(cells, d.nx, d.ny, false);
        ops.exp_grid_backprop(cells, derivs, d.nx, d.ny);
        ops.sync();
        ops.prefix(cells, d.nx, d.ny, true);
    } else {
        const T seed = exp(ops.sum(local))*(*derivs);
        for (uint64_t k = ops.index(); k < d.cells; k += ops.stride()) cells[k] = seed;
    }
    ops.sync();
    for (uint64_t k = ops.index(); k < d.cells; k += ops.stride()) {
        const uint64_t i = k/d.ny, j = k%d.ny;
        T value = 0;
        for (uint64_t r = 0; r < d.q; ++r)
            for (uint64_t s = 0; s < d.q; ++s) {
                const uint64_t v = (i*d.q+r)*d.width+j*d.q+s;
                const T seed = cells[k]*weights[r]*weights[s];
                value += seed*previous[v];
                adjoint[v] = seed*gram[k];
            }
        grad[k] = value;
    }
    ops.sync();
    for (uint64_t m = depth-1; m > 0; --m) {
        const T* current = state + m*d.nodes;
        previous = current - d.nodes;
        for (uint64_t k = ops.index(); k < d.nodes; k += ops.stride()) adjoint[k] *= current[k];
        ops.sync();
        ops.integrate_y(adjoint, work.integrated, d.height, d.ny, d.q, tables, weights, true);
        ops.integrate_x(work.integrated, adjoint, d.nx, d.width, d.q, tables, weights, true);
        for (uint64_t k = ops.index(); k < d.cells; k += ops.stride()) {
            const uint64_t i = k/d.ny, j = k%d.ny;
            T value = 0;
            for (uint64_t r = 0; r < d.q; ++r)
                for (uint64_t s = 0; s < d.q; ++s) {
                    const uint64_t v = (i*d.q+r)*d.width+j*d.q+s;
                    value += adjoint[v]*previous[v];
                    adjoint[v] *= gram[k];
                }
            grad[k] += value;
        }
        ops.sync();
    }
}
#undef BSK_DEVICE
