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

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cp_exp_signature.h"
#include "cp_tensor_log.h"
#include "cp_vector_funcs.h"
#include "macros.h"
#include "multithreading.h"

namespace log_pde_detail {

struct Layout {
	uint64_t dimension;
	uint64_t degree_x;
	uint64_t degree_y;
	uint64_t degree_f;
	uint64_t degree_g;
	std::vector<uint64_t> offset;
	uint64_t x_len;
	uint64_t y_len;
	uint64_t f_len;
	uint64_t g_len;
	uint64_t state_len;
	uint64_t cell_cache_len;

	Layout(uint64_t dimension_, uint64_t degree_x_, uint64_t degree_y_)
		: dimension(dimension_), degree_x(degree_x_), degree_y(degree_y_),
		degree_f(degree_y_ - 1), degree_g(degree_x_ - 1) {
		if (dimension == 0) throw std::invalid_argument("log-PDE dimension must be positive");
		if (degree_x == 0 || degree_y == 0)
			throw std::invalid_argument("log-PDE degrees must be positive");

		const uint64_t max_degree = std::max(degree_x, degree_y);
		offset.assign(max_degree + 2, 0);
		uint64_t level_size = 1;
		for (uint64_t level = 1; level <= max_degree; ++level) {
			if (level_size > std::numeric_limits<uint64_t>::max() / dimension)
				throw std::overflow_error("log-PDE tensor length overflow");
			level_size *= dimension;
			if (offset[level] > std::numeric_limits<uint64_t>::max() - level_size)
				throw std::overflow_error("log-PDE tensor length overflow");
			offset[level + 1] = offset[level] + level_size;
		}

		x_len = length(degree_x);
		y_len = length(degree_y);
		f_len = length(degree_f);
		g_len = length(degree_g);
		state_len = 1 + f_len + g_len;
		cell_cache_len = 2 * (f_len + g_len) + 3;
	}

	uint64_t length(uint64_t degree) const {
		return degree == 0 ? 0 : offset[degree + 1];
	}

	uint64_t level_size(uint64_t level) const {
		return offset[level + 1] - offset[level];
	}
};

inline uint64_t refinement(uint64_t order) {
	if (order >= 63) throw std::invalid_argument("log-PDE dyadic order is too large");
	return 1ULL << order;
}

inline uint64_t fine_steps(uint64_t steps, uint64_t factor) {
	if (steps == 0) throw std::invalid_argument("log-PDE requires at least one log increment");
	if (steps > (std::numeric_limits<uint64_t>::max() - 1) / factor)
		throw std::overflow_error("log-PDE grid length overflow");
	return steps * factor;
}

template<std::floating_point T>
void tensor_adjoint_left_add(
	const Layout& layout,
	const T* RESTRICT w,
	uint64_t w_degree,
	const T* RESTRICT y,
	uint64_t y_degree,
	T* RESTRICT out,
	uint64_t out_degree
) {
	for (uint64_t level = 1; level <= out_degree; ++level) {
		const uint64_t out_size = layout.level_size(level);
		T* dst = out + layout.offset[level];
		if (level >= y_degree) continue;
		const uint64_t last = std::min(w_degree, y_degree - level);
		for (uint64_t wi = 1; wi <= last; ++wi) {
			const uint64_t w_size = layout.level_size(wi);
			const T* wp = w + layout.offset[wi];
			const T* yp = y + layout.offset[level + wi];
			for (uint64_t i = 0; i < w_size; ++i) {
				const T wi_value = wp[i];
				const T* row = yp + i * out_size;
				for (uint64_t j = 0; j < out_size; ++j)
					dst[j] += wi_value * row[j];
			}
		}
	}
}

template<std::floating_point T>
void tensor_adjoint_left_backward(
	const Layout& layout,
	const T* RESTRICT w,
	uint64_t w_degree,
	const T* RESTRICT y,
	uint64_t y_degree,
	const T* RESTRICT d_out,
	uint64_t out_degree,
	T* RESTRICT d_w,
	T* RESTRICT d_y
) {
	for (uint64_t level = 1; level <= out_degree; ++level) {
		const uint64_t out_size = layout.level_size(level);
		const T* grad = d_out + layout.offset[level];
		for (uint64_t wi = 1; wi <= w_degree && level + wi <= y_degree; ++wi) {
			const uint64_t w_size = layout.level_size(wi);
			const T* wp = w + layout.offset[wi];
			const T* yp = y + layout.offset[level + wi];
			T* dwp = d_w + layout.offset[wi];
			T* dyp = d_y + layout.offset[level + wi];
			for (uint64_t i = 0; i < w_size; ++i) {
				T dw = static_cast<T>(0);
				const T wi_value = wp[i];
				const T* row = yp + i * out_size;
				T* d_row = dyp + i * out_size;
				for (uint64_t j = 0; j < out_size; ++j)
					dw += grad[j] * row[j];
				dwp[i] += dw;
				for (uint64_t j = 0; j < out_size; ++j)
					d_row[j] += wi_value * grad[j];
			}
		}
	}
}

template<std::floating_point T>
void tensor_adjoint_right_add(
	const Layout& layout,
	const T* w,
	uint64_t w_degree,
	const T* y,
	uint64_t y_degree,
	T* out,
	uint64_t out_degree
) {
	for (uint64_t level = 1; level <= out_degree; ++level) {
		const uint64_t out_size = layout.level_size(level);
		T* dst = out + layout.offset[level];
		for (uint64_t wi = 1; wi <= w_degree && level + wi <= y_degree; ++wi) {
			const uint64_t w_size = layout.level_size(wi);
			const T* wp = w + layout.offset[wi];
			const T* yp = y + layout.offset[level + wi];
			for (uint64_t i = 0; i < out_size; ++i) {
				T sum = static_cast<T>(0);
				for (uint64_t j = 0; j < w_size; ++j)
					sum += yp[i * w_size + j] * wp[j];
				dst[i] += sum;
			}
		}
	}
}

template<std::floating_point T>
void tensor_adjoint_right_backward(
	const Layout& layout,
	const T* w,
	uint64_t w_degree,
	const T* y,
	uint64_t y_degree,
	const T* d_out,
	uint64_t out_degree,
	T* d_w,
	T* d_y
) {
	for (uint64_t level = 1; level <= out_degree; ++level) {
		const uint64_t out_size = layout.level_size(level);
		const T* grad = d_out + layout.offset[level];
		for (uint64_t wi = 1; wi <= w_degree && level + wi <= y_degree; ++wi) {
			const uint64_t w_size = layout.level_size(wi);
			const T* wp = w + layout.offset[wi];
			const T* yp = y + layout.offset[level + wi];
			T* dwp = d_w + layout.offset[wi];
			T* dyp = d_y + layout.offset[level + wi];
			for (uint64_t i = 0; i < out_size; ++i) {
				for (uint64_t j = 0; j < w_size; ++j) {
					const T dg = grad[i];
					dwp[j] += yp[i * w_size + j] * dg;
					dyp[i * w_size + j] += dg * wp[j];
				}
			}
		}
	}
}

template<std::floating_point T>
struct CellScratch {
	T gamma;
	T u_base;
	T u_provisional;
	std::vector<T> dx_adj_dy;
	std::vector<T> dy_adj_dx;
	std::vector<T> df0;
	std::vector<T> df1;
	std::vector<T> fp;
	std::vector<T> dg0;
	std::vector<T> dg1;
	std::vector<T> gp;
	std::vector<T> ad_fse;
	std::vector<T> ad_gse;
	std::vector<T> ad_df0;
	std::vector<T> ad_df1;
	std::vector<T> ad_fp;
	std::vector<T> ad_dg0;
	std::vector<T> ad_dg1;
	std::vector<T> ad_gp;
	std::vector<T> ad_dx_adj_dy;
	std::vector<T> ad_dy_adj_dx;
	std::vector<T> d_dx;
	std::vector<T> d_dy;

	void resize(const Layout& layout) {
		dx_adj_dy.resize(layout.f_len);
		dy_adj_dx.resize(layout.g_len);
		df0.resize(layout.f_len);
		df1.resize(layout.f_len);
		fp.resize(layout.f_len);
		dg0.resize(layout.g_len);
		dg1.resize(layout.g_len);
		gp.resize(layout.g_len);
		ad_fse.resize(layout.f_len);
		ad_gse.resize(layout.g_len);
		ad_df0.resize(layout.f_len);
		ad_df1.resize(layout.f_len);
		ad_fp.resize(layout.f_len);
		ad_dg0.resize(layout.g_len);
		ad_dg1.resize(layout.g_len);
		ad_gp.resize(layout.g_len);
		ad_dx_adj_dy.resize(layout.f_len);
		ad_dy_adj_dx.resize(layout.g_len);
		d_dx.resize(layout.x_len);
		d_dy.resize(layout.y_len);
	}
};

template<std::floating_point T>
void f_increment(
	const Layout& layout,
	const T* dx,
	T u,
	const T* f,
	const T* g,
	T* out
) {
	std::fill(out, out + layout.f_len, static_cast<T>(0));
	const uint64_t drive_degree = std::min(layout.degree_x, layout.degree_f);
	const uint64_t drive_length = layout.length(drive_degree);
	for (uint64_t i = 0; i < drive_length; ++i) out[i] += u * dx[i];
	tensor_product_add_(f, layout.degree_f, dx, drive_degree, out, layout.degree_f,
		layout.offset.data());
	tensor_adjoint_left_add(layout, g, layout.degree_g, dx, layout.degree_x,
		out, layout.degree_f);
}

template<std::floating_point T>
void g_increment(
	const Layout& layout,
	const T* dy,
	T u,
	const T* f,
	const T* g,
	T* out
) {
	std::fill(out, out + layout.g_len, static_cast<T>(0));
	const uint64_t drive_degree = std::min(layout.degree_y, layout.degree_g);
	const uint64_t drive_length = layout.length(drive_degree);
	for (uint64_t i = 0; i < drive_length; ++i) out[i] += u * dy[i];
	tensor_product_add_(g, layout.degree_g, dy, drive_degree, out, layout.degree_g,
		layout.offset.data());
	tensor_adjoint_left_add(layout, f, layout.degree_f, dy, layout.degree_y,
		out, layout.degree_g);
}

template<std::floating_point T>
void f_increment_backward(
	const Layout& layout,
	const T* dx,
	T u,
	const T* f,
	const T* g,
	const T* d_out,
	T& d_u,
	T* d_f,
	T* d_g,
	T* d_dx
) {
	const uint64_t drive_degree = std::min(layout.degree_x, layout.degree_f);
	const uint64_t drive_length = layout.length(drive_degree);
	d_u += dot_product(dx, d_out, drive_length);
	for (uint64_t i = 0; i < drive_length; ++i) d_dx[i] += u * d_out[i];
	tensor_product_backprop_(f, layout.degree_f, dx, drive_degree,
		d_out, layout.degree_f, d_f, d_dx, layout.offset.data());
	tensor_adjoint_left_backward(layout, g, layout.degree_g, dx, layout.degree_x,
		d_out, layout.degree_f, d_g, d_dx);
}

template<std::floating_point T>
void g_increment_backward(
	const Layout& layout,
	const T* dy,
	T u,
	const T* f,
	const T* g,
	const T* d_out,
	T& d_u,
	T* d_f,
	T* d_g,
	T* d_dy
) {
	const uint64_t drive_degree = std::min(layout.degree_y, layout.degree_g);
	const uint64_t drive_length = layout.length(drive_degree);
	d_u += dot_product(dy, d_out, drive_length);
	for (uint64_t i = 0; i < drive_length; ++i) d_dy[i] += u * d_out[i];
	tensor_product_backprop_(g, layout.degree_g, dy, drive_degree,
		d_out, layout.degree_g, d_g, d_dy, layout.offset.data());
	tensor_adjoint_left_backward(layout, f, layout.degree_f, dy, layout.degree_y,
		d_out, layout.degree_g, d_f, d_dy);
}

template<std::floating_point T>
T forcing(const Layout& layout, T u, const T* f, const T* g,
	T gamma, const T* dx_adj_dy, const T* dy_adj_dx) {
	return u * gamma + dot_product(f, dx_adj_dy, layout.f_len)
		+ dot_product(g, dy_adj_dx, layout.g_len);
}

template<std::floating_point T>
void forcing_backward(
	const Layout& layout,
	T d_out,
	T u,
	const T* f,
	const T* g,
	T gamma,
	const T* dx_adj_dy,
	const T* dy_adj_dx,
	T& d_u,
	T* d_f,
	T* d_g,
	T& d_gamma,
	T* d_dx_adj_dy,
	T* d_dy_adj_dx
) {
	d_u += d_out * gamma;
	d_gamma += d_out * u;
	for (uint64_t i = 0; i < layout.f_len; ++i) {
		d_f[i] += d_out * dx_adj_dy[i];
		d_dx_adj_dy[i] += d_out * f[i];
	}
	for (uint64_t i = 0; i < layout.g_len; ++i) {
		d_g[i] += d_out * dy_adj_dx[i];
		d_dy_adj_dx[i] += d_out * g[i];
	}
}

template<std::floating_point T>
void cell_forward(
	const Layout& layout,
	const T* dx,
	const T* dy,
	const T* nw,
	const T* north,
	const T* west,
	T* se,
	const T* known_u,
	CellScratch<T>& scratch
) {
	if (layout.degree_x == 1 && layout.degree_y == 1) {
		scratch.gamma = dot_product(dx, dy, layout.x_len);
		scratch.u_base = north[0] + west[0] - nw[0];
		scratch.u_provisional = static_cast<T>(0);
		if (known_u != nullptr) {
			se[0] = known_u[0];
			return;
		}
		constexpr T kappa = static_cast<T>(1.0 / 12.0);
		const T gamma_squared = scratch.gamma * scratch.gamma;
		se[0] = (north[0] + west[0]) *
			(static_cast<T>(1) + static_cast<T>(0.5) * scratch.gamma + kappa * gamma_squared)
			- nw[0] * (static_cast<T>(1) - kappa * gamma_squared);
		return;
	}

	const T* f_nw = nw + 1;
	const T* f_n = north + 1;
	const T* f_w = west + 1;
	const T* g_nw = f_nw + layout.f_len;
	const T* g_n = f_n + layout.f_len;
	const T* g_w = f_w + layout.f_len;
	T* f_se = se + 1;
	T* g_se = f_se + layout.f_len;

	std::fill(scratch.dx_adj_dy.begin(), scratch.dx_adj_dy.end(), static_cast<T>(0));
	std::fill(scratch.dy_adj_dx.begin(), scratch.dy_adj_dx.end(), static_cast<T>(0));
	tensor_adjoint_right_add(layout, dx, layout.degree_x, dy, layout.degree_y,
		scratch.dx_adj_dy.data(), layout.degree_f);
	tensor_adjoint_right_add(layout, dy, layout.degree_y, dx, layout.degree_x,
		scratch.dy_adj_dx.data(), layout.degree_g);
	const uint64_t common_degree = std::min(layout.degree_x, layout.degree_y);
	scratch.gamma = dot_product(dx, dy, layout.length(common_degree));

	f_increment(layout, dx, north[0], f_n, g_n, scratch.df0.data());
	g_increment(layout, dy, west[0], f_w, g_w, scratch.dg0.data());
	for (uint64_t i = 0; i < layout.f_len; ++i) scratch.fp[i] = f_n[i] + scratch.df0[i];
	for (uint64_t i = 0; i < layout.g_len; ++i) scratch.gp[i] = g_w[i] + scratch.dg0[i];

	const T f_nw_value = forcing(layout, nw[0], f_nw, g_nw, scratch.gamma,
		scratch.dx_adj_dy.data(), scratch.dy_adj_dx.data());
	scratch.u_base = north[0] + west[0] - nw[0];
	scratch.u_provisional = scratch.u_base + f_nw_value;

	f_increment(layout, dx, scratch.u_provisional, scratch.fp.data(), scratch.gp.data(), scratch.df1.data());
	g_increment(layout, dy, scratch.u_provisional, scratch.fp.data(), scratch.gp.data(), scratch.dg1.data());
	for (uint64_t i = 0; i < layout.f_len; ++i)
		f_se[i] = f_n[i] + static_cast<T>(0.5) * (scratch.df0[i] + scratch.df1[i]);
	for (uint64_t i = 0; i < layout.g_len; ++i)
		g_se[i] = g_w[i] + static_cast<T>(0.5) * (scratch.dg0[i] + scratch.dg1[i]);

	if (known_u != nullptr) {
		se[0] = known_u[0];
		return;
	}
	const T f_n_value = forcing(layout, north[0], f_n, g_n, scratch.gamma,
		scratch.dx_adj_dy.data(), scratch.dy_adj_dx.data());
	const T f_w_value = forcing(layout, west[0], f_w, g_w, scratch.gamma,
		scratch.dx_adj_dy.data(), scratch.dy_adj_dx.data());
	const T f_corner = forcing(layout, scratch.u_provisional, f_se, g_se, scratch.gamma,
		scratch.dx_adj_dy.data(), scratch.dy_adj_dx.data());
	se[0] = scratch.u_base + static_cast<T>(0.25) *
		(f_nw_value + f_n_value + f_w_value + f_corner);
}

template<std::floating_point T>
void store_cell_cache(
	const Layout& layout,
	const CellScratch<T>& scratch,
	T* cache
) {
	cache = std::copy(scratch.dx_adj_dy.begin(), scratch.dx_adj_dy.end(), cache);
	cache = std::copy(scratch.dy_adj_dx.begin(), scratch.dy_adj_dx.end(), cache);
	cache = std::copy(scratch.fp.begin(), scratch.fp.end(), cache);
	cache = std::copy(scratch.gp.begin(), scratch.gp.end(), cache);
	cache[0] = scratch.gamma;
	cache[1] = scratch.u_base;
	cache[2] = scratch.u_provisional;
}

template<std::floating_point T>
void cell_backward(
	const Layout& layout,
	const T* dx,
	const T* dy,
	const T* nw,
	const T* north,
	const T* west,
	const T* se,
	const T* cache,
	const T* d_se,
	T* d_nw,
	T* d_north,
	T* d_west,
	CellScratch<T>& scratch
) {
	if (layout.degree_x == 1 && layout.degree_y == 1) {
		constexpr T kappa = static_cast<T>(1.0 / 12.0);
		const T gamma = cache[0];
		const T gamma_squared = gamma * gamma;
		const T d_value = d_se[0];
		const T edge_coefficient = static_cast<T>(1) + static_cast<T>(0.5) * gamma
			+ kappa * gamma_squared;
		d_north[0] += d_value * edge_coefficient;
		d_west[0] += d_value * edge_coefficient;
		d_nw[0] -= d_value * (static_cast<T>(1) - kappa * gamma_squared);
		const T d_gamma = d_value * (
			(north[0] + west[0]) * (static_cast<T>(0.5) + gamma / static_cast<T>(6))
			+ nw[0] * gamma / static_cast<T>(6)
		);
		for (uint64_t i = 0; i < layout.x_len; ++i) {
			scratch.d_dx[i] = d_gamma * dy[i];
			scratch.d_dy[i] = d_gamma * dx[i];
		}
		return;
	}

	const T* f_nw = nw + 1;
	const T* f_n = north + 1;
	const T* f_w = west + 1;
	const T* g_nw = f_nw + layout.f_len;
	const T* g_n = f_n + layout.f_len;
	const T* g_w = f_w + layout.f_len;
	const T* f_se = se + 1;
	const T* g_se = f_se + layout.f_len;
	const T* dx_adj_dy = cache;
	const T* dy_adj_dx = dx_adj_dy + layout.f_len;
	const T* fp = dy_adj_dx + layout.g_len;
	const T* gp = fp + layout.f_len;
	const T* scalars = gp + layout.g_len;
	T* d_f_nw = d_nw + 1;
	T* d_f_n = d_north + 1;
	T* d_f_w = d_west + 1;
	T* d_g_nw = d_f_nw + layout.f_len;
	T* d_g_n = d_f_n + layout.f_len;
	T* d_g_w = d_f_w + layout.f_len;

	std::fill(scratch.ad_fse.begin(), scratch.ad_fse.end(), static_cast<T>(0));
	std::fill(scratch.ad_gse.begin(), scratch.ad_gse.end(), static_cast<T>(0));
	std::copy(d_se + 1, d_se + 1 + layout.f_len, scratch.ad_fse.begin());
	std::copy(d_se + 1 + layout.f_len, d_se + layout.state_len, scratch.ad_gse.begin());
	std::fill(scratch.ad_df0.begin(), scratch.ad_df0.end(), static_cast<T>(0));
	std::fill(scratch.ad_df1.begin(), scratch.ad_df1.end(), static_cast<T>(0));
	std::fill(scratch.ad_fp.begin(), scratch.ad_fp.end(), static_cast<T>(0));
	std::fill(scratch.ad_dg0.begin(), scratch.ad_dg0.end(), static_cast<T>(0));
	std::fill(scratch.ad_dg1.begin(), scratch.ad_dg1.end(), static_cast<T>(0));
	std::fill(scratch.ad_gp.begin(), scratch.ad_gp.end(), static_cast<T>(0));
	std::fill(scratch.ad_dx_adj_dy.begin(), scratch.ad_dx_adj_dy.end(), static_cast<T>(0));
	std::fill(scratch.ad_dy_adj_dx.begin(), scratch.ad_dy_adj_dx.end(), static_cast<T>(0));
	std::fill(scratch.d_dx.begin(), scratch.d_dx.end(), static_cast<T>(0));
	std::fill(scratch.d_dy.begin(), scratch.d_dy.end(), static_cast<T>(0));

	const uint64_t common_degree = std::min(layout.degree_x, layout.degree_y);
	const uint64_t common_length = layout.length(common_degree);
	const T gamma = scalars[0];
	const T u_base = scalars[1];
	const T u_provisional = scalars[2];

	T d_u_base = d_se[0];
	T d_f_nw_value = static_cast<T>(0.25) * d_se[0];
	T d_f_n_value = static_cast<T>(0.25) * d_se[0];
	T d_f_w_value = static_cast<T>(0.25) * d_se[0];
	const T d_f_corner = static_cast<T>(0.25) * d_se[0];
	T d_u_provisional = static_cast<T>(0);
	T d_gamma = static_cast<T>(0);

	forcing_backward(layout, d_f_corner, u_provisional, f_se, g_se, gamma,
		dx_adj_dy, dy_adj_dx,
		d_u_provisional, scratch.ad_fse.data(), scratch.ad_gse.data(), d_gamma,
		scratch.ad_dx_adj_dy.data(), scratch.ad_dy_adj_dx.data());

	for (uint64_t i = 0; i < layout.f_len; ++i) {
		d_f_n[i] += scratch.ad_fse[i];
		scratch.ad_df0[i] += static_cast<T>(0.5) * scratch.ad_fse[i];
		scratch.ad_df1[i] += static_cast<T>(0.5) * scratch.ad_fse[i];
	}
	for (uint64_t i = 0; i < layout.g_len; ++i) {
		d_g_w[i] += scratch.ad_gse[i];
		scratch.ad_dg0[i] += static_cast<T>(0.5) * scratch.ad_gse[i];
		scratch.ad_dg1[i] += static_cast<T>(0.5) * scratch.ad_gse[i];
	}

	f_increment_backward(layout, dx, u_provisional, fp, gp,
		scratch.ad_df1.data(), d_u_provisional, scratch.ad_fp.data(), scratch.ad_gp.data(),
		scratch.d_dx.data());
	g_increment_backward(layout, dy, u_provisional, fp, gp,
		scratch.ad_dg1.data(), d_u_provisional, scratch.ad_fp.data(), scratch.ad_gp.data(),
		scratch.d_dy.data());

	for (uint64_t i = 0; i < layout.f_len; ++i) {
		d_f_n[i] += scratch.ad_fp[i];
		scratch.ad_df0[i] += scratch.ad_fp[i];
	}
	for (uint64_t i = 0; i < layout.g_len; ++i) {
		d_g_w[i] += scratch.ad_gp[i];
		scratch.ad_dg0[i] += scratch.ad_gp[i];
	}

	d_u_base += d_u_provisional;
	d_f_nw_value += d_u_provisional;
	forcing_backward(layout, d_f_nw_value, nw[0], f_nw, g_nw, gamma,
		dx_adj_dy, dy_adj_dx,
		d_nw[0], d_f_nw, d_g_nw, d_gamma,
		scratch.ad_dx_adj_dy.data(), scratch.ad_dy_adj_dx.data());
	forcing_backward(layout, d_f_n_value, north[0], f_n, g_n, gamma,
		dx_adj_dy, dy_adj_dx,
		d_north[0], d_f_n, d_g_n, d_gamma,
		scratch.ad_dx_adj_dy.data(), scratch.ad_dy_adj_dx.data());
	forcing_backward(layout, d_f_w_value, west[0], f_w, g_w, gamma,
		dx_adj_dy, dy_adj_dx,
		d_west[0], d_f_w, d_g_w, d_gamma,
		scratch.ad_dx_adj_dy.data(), scratch.ad_dy_adj_dx.data());

	f_increment_backward(layout, dx, north[0], f_n, g_n, scratch.ad_df0.data(),
		d_north[0], d_f_n, d_g_n, scratch.d_dx.data());
	g_increment_backward(layout, dy, west[0], f_w, g_w, scratch.ad_dg0.data(),
		d_west[0], d_f_w, d_g_w, scratch.d_dy.data());

	d_north[0] += d_u_base;
	d_west[0] += d_u_base;
	d_nw[0] -= d_u_base;

	for (uint64_t i = 0; i < common_length; ++i) {
		scratch.d_dx[i] += d_gamma * dy[i];
		scratch.d_dy[i] += d_gamma * dx[i];
	}
	tensor_adjoint_right_backward(layout, dx, layout.degree_x, dy, layout.degree_y,
		scratch.ad_dx_adj_dy.data(), layout.degree_f,
		scratch.d_dx.data(), scratch.d_dy.data());
	tensor_adjoint_right_backward(layout, dy, layout.degree_y, dx, layout.degree_x,
		scratch.ad_dy_adj_dx.data(), layout.degree_g,
		scratch.d_dy.data(), scratch.d_dx.data());
}

template<std::floating_point T>
struct Workspace {
	std::vector<T> scaled_x;
	std::vector<T> scaled_y;
	std::vector<T> boundary_x;
	std::vector<T> boundary_y;
	std::vector<T> row_a;
	std::vector<T> row_b;
	std::vector<T> tape;
	std::vector<T> cell_cache;
	std::vector<T> adjoint;
	std::vector<T> exp_value;
	std::vector<T> boundary_next;
	std::vector<T> powers;
	std::vector<T> d_powers;
	std::vector<T> temp;
	std::vector<T> coarse_adj;
	std::vector<T> d_current;
	std::vector<T> d_previous;
	std::vector<T> d_exp;
	CellScratch<T> cell;

	void prepare(const Layout& layout, uint64_t steps_x, uint64_t steps_y,
		uint64_t nodes_x, uint64_t nodes_y, bool need_tape) {
		scaled_x.resize(steps_x * layout.x_len);
		scaled_y.resize(steps_y * layout.y_len);
		boundary_x.resize((steps_x + 1) * layout.f_len);
		boundary_y.resize((steps_y + 1) * layout.g_len);
		row_a.resize(nodes_y * layout.state_len);
		row_b.resize(nodes_y * layout.state_len);
		if (need_tape) {
			tape.resize(nodes_x * nodes_y * layout.state_len);
			cell_cache.resize((nodes_x - 1) * (nodes_y - 1) * layout.cell_cache_len);
			adjoint.resize(nodes_x * nodes_y * layout.state_len);
		}
		cell.resize(layout);
	}
};

template<std::floating_point T>
void build_boundary(
	const Layout& layout,
	const T* logs,
	uint64_t steps,
	uint64_t log_degree,
	uint64_t boundary_degree,
	std::vector<T>& boundary,
	Workspace<T>& workspace
) {
	const uint64_t length = layout.length(boundary_degree);
	if (length == 0) return;
	std::fill(boundary.begin(), boundary.end(), static_cast<T>(0));
	workspace.exp_value.resize(length);
	workspace.boundary_next.resize(length);
	const uint64_t log_stride = layout.length(log_degree);
	for (uint64_t step = 0; step < steps; ++step) {
		tensor_exp_with_level_index_(logs + step * log_stride, workspace.exp_value.data(),
			log_degree, boundary_degree, layout.offset.data(),
			workspace.powers, workspace.temp);
		sig_combine_with_level_index_(boundary.data() + step * length,
			workspace.exp_value.data(), workspace.boundary_next.data(), boundary_degree,
			layout.offset.data(), false);
		std::copy(workspace.boundary_next.begin(), workspace.boundary_next.end(),
			boundary.begin() + (step + 1) * length);
	}
}

template<std::floating_point T>
void prepare_forward_data(
	const Layout& layout,
	const T* logs_x,
	const T* logs_y,
	uint64_t steps_x,
	uint64_t steps_y,
	uint64_t factor_x,
	uint64_t factor_y,
	Workspace<T>& workspace
) {
	for (uint64_t i = 0; i < steps_x * layout.x_len; ++i)
		workspace.scaled_x[i] = logs_x[i] / static_cast<T>(factor_x);
	for (uint64_t i = 0; i < steps_y * layout.y_len; ++i)
		workspace.scaled_y[i] = logs_y[i] / static_cast<T>(factor_y);
	build_boundary(layout, logs_x, steps_x, layout.degree_x, layout.degree_f,
		workspace.boundary_x, workspace);
	build_boundary(layout, logs_y, steps_y, layout.degree_y, layout.degree_g,
		workspace.boundary_y, workspace);
}

template<std::floating_point T>
void solve_forward(
	const Layout& layout,
	const T* logs_x,
	const T* logs_y,
	T* out,
	uint64_t steps_x,
	uint64_t steps_y,
	uint64_t factor_x,
	uint64_t factor_y,
	bool return_grid,
	Workspace<T>& workspace
) {
	const uint64_t fine_x = steps_x * factor_x;
	const uint64_t fine_y = steps_y * factor_y;
	const uint64_t nodes_x = fine_x + 1;
	const uint64_t nodes_y = fine_y + 1;
	workspace.prepare(layout, steps_x, steps_y, nodes_x, nodes_y, false);
	prepare_forward_data(layout, logs_x, logs_y, steps_x, steps_y, factor_x, factor_y, workspace);
	std::fill(workspace.row_a.begin(), workspace.row_a.end(), static_cast<T>(0));
	for (uint64_t j = 0; j < nodes_y; ++j) {
		T* state = workspace.row_a.data() + j * layout.state_len;
		state[0] = static_cast<T>(1);
		const uint64_t coarse = std::min(j / factor_y, steps_y);
		if (layout.g_len != 0)
			std::copy(workspace.boundary_y.begin() + coarse * layout.g_len,
				workspace.boundary_y.begin() + (coarse + 1) * layout.g_len,
				state + 1 + layout.f_len);
		if (return_grid) out[j] = static_cast<T>(1);
	}

	for (uint64_t i = 1; i < nodes_x; ++i) {
		std::fill(workspace.row_b.begin(), workspace.row_b.end(), static_cast<T>(0));
		T* west = workspace.row_b.data();
		west[0] = static_cast<T>(1);
		const uint64_t coarse_x = std::min(i / factor_x, steps_x);
		if (layout.f_len != 0)
			std::copy(workspace.boundary_x.begin() + coarse_x * layout.f_len,
				workspace.boundary_x.begin() + (coarse_x + 1) * layout.f_len, west + 1);
		if (return_grid) out[i * nodes_y] = static_cast<T>(1);

		const T* dx = workspace.scaled_x.data() + ((i - 1) / factor_x) * layout.x_len;
		for (uint64_t j = 1; j < nodes_y; ++j) {
			const T* dy = workspace.scaled_y.data() + ((j - 1) / factor_y) * layout.y_len;
			T* se = workspace.row_b.data() + j * layout.state_len;
			cell_forward(layout, dx, dy,
				workspace.row_a.data() + (j - 1) * layout.state_len,
				workspace.row_a.data() + j * layout.state_len,
				workspace.row_b.data() + (j - 1) * layout.state_len,
				se, static_cast<const T*>(nullptr), workspace.cell);
			if (return_grid) out[i * nodes_y + j] = se[0];
		}
		workspace.row_a.swap(workspace.row_b);
	}
	if (!return_grid) out[0] = workspace.row_a[(nodes_y - 1) * layout.state_len];
}

template<std::floating_point T>
void build_path_logs(
	const T* path,
	T* logs,
	uint64_t dimension,
	uint64_t steps,
	uint64_t log_step,
	uint64_t degree,
	std::vector<T>& signature
) {
	const uint64_t full_length = ::sig_length(dimension, degree);
	signature.resize(full_length);
	for (uint64_t step = 0; step < steps; ++step) {
		signature_<T>(
			path + step * log_step * dimension, signature.data(),
			1, dimension, log_step + 1, degree
		);
		tensor_log_<T>(signature.data(), dimension, degree);
		std::copy(signature.begin() + 1, signature.end(),
			logs + step * (full_length - 1));
	}
}

template<std::floating_point T>
void path_logs_backward(
	const T* path,
	const T* d_logs,
	T* d_path,
	uint64_t dimension,
	uint64_t length,
	uint64_t steps,
	uint64_t log_step,
	uint64_t degree,
	std::vector<T>& signature,
	std::vector<T>& log_derivs,
	std::vector<T>& sig_derivs,
	std::vector<T>& block_derivs
) {
	const uint64_t full_length = ::sig_length(dimension, degree);
	const uint64_t log_length = full_length - 1;
	signature.resize(full_length);
	log_derivs.resize(full_length);
	sig_derivs.resize(full_length);
	block_derivs.resize((log_step + 1) * dimension);
	std::fill(d_path, d_path + length * dimension, static_cast<T>(0));

	for (uint64_t step = 0; step < steps; ++step) {
		const T* block = path + step * log_step * dimension;
		signature_<T>(block, signature.data(), 1, dimension, log_step + 1, degree);
		log_derivs[0] = static_cast<T>(0);
		std::copy(d_logs + step * log_length, d_logs + (step + 1) * log_length,
			log_derivs.begin() + 1);
		tensor_log_backprop_<T>(
			sig_derivs.data(), log_derivs.data(), signature.data(), dimension, degree
		);
		std::fill(block_derivs.begin(), block_derivs.end(), static_cast<T>(0));
		Path<T> block_path(block, dimension, log_step + 1);
		sig_backprop_inplace_<T>(
			block_path, block_derivs.data(), sig_derivs.data(), signature.data(),
			degree, full_length
		);
		T* d_block = d_path + step * log_step * dimension;
		for (uint64_t i = 0; i < (log_step + 1) * dimension; ++i)
			d_block[i] += block_derivs[i];
	}
}

template<std::floating_point T>
void build_tape(
	const Layout& layout,
	uint64_t steps_x,
	uint64_t steps_y,
	uint64_t factor_x,
	uint64_t factor_y,
	const T* k_grid,
	Workspace<T>& workspace
) {
	const uint64_t nodes_x = steps_x * factor_x + 1;
	const uint64_t nodes_y = steps_y * factor_y + 1;
	std::fill(workspace.tape.begin(), workspace.tape.end(), static_cast<T>(0));
	for (uint64_t i = 0; i < nodes_x; ++i) {
		T* state = workspace.tape.data() + i * nodes_y * layout.state_len;
		state[0] = static_cast<T>(1);
		const uint64_t coarse = std::min(i / factor_x, steps_x);
		if (layout.f_len != 0)
			std::copy(workspace.boundary_x.begin() + coarse * layout.f_len,
				workspace.boundary_x.begin() + (coarse + 1) * layout.f_len, state + 1);
	}
	for (uint64_t j = 0; j < nodes_y; ++j) {
		T* state = workspace.tape.data() + j * layout.state_len;
		state[0] = static_cast<T>(1);
		const uint64_t coarse = std::min(j / factor_y, steps_y);
		if (layout.g_len != 0)
			std::copy(workspace.boundary_y.begin() + coarse * layout.g_len,
				workspace.boundary_y.begin() + (coarse + 1) * layout.g_len,
				state + 1 + layout.f_len);
	}

	for (uint64_t i = 1; i < nodes_x; ++i) {
		const T* dx = workspace.scaled_x.data() + ((i - 1) / factor_x) * layout.x_len;
		for (uint64_t j = 1; j < nodes_y; ++j) {
			const T* dy = workspace.scaled_y.data() + ((j - 1) / factor_y) * layout.y_len;
			const uint64_t node = i * nodes_y + j;
			T* se = workspace.tape.data() + node * layout.state_len;
			cell_forward(layout, dx, dy,
				workspace.tape.data() + ((i - 1) * nodes_y + j - 1) * layout.state_len,
				workspace.tape.data() + ((i - 1) * nodes_y + j) * layout.state_len,
				workspace.tape.data() + (i * nodes_y + j - 1) * layout.state_len,
				se, k_grid == nullptr ? nullptr : k_grid + node, workspace.cell);
			store_cell_cache(layout, workspace.cell,
				workspace.cell_cache.data() +
				((i - 1) * (nodes_y - 1) + j - 1) * layout.cell_cache_len);
		}
	}
}

template<std::floating_point T>
void boundary_backward(
	const Layout& layout,
	const T* logs,
	uint64_t steps,
	uint64_t log_degree,
	uint64_t boundary_degree,
	const std::vector<T>& boundary,
	const std::vector<T>& coarse_adjoint,
	T* d_logs,
	Workspace<T>& workspace
) {
	const uint64_t length = layout.length(boundary_degree);
	if (length == 0) return;
	const uint64_t log_stride = layout.length(log_degree);
	workspace.d_current.assign(coarse_adjoint.begin() + steps * length,
		coarse_adjoint.begin() + (steps + 1) * length);
	workspace.d_previous.resize(length);
	workspace.d_exp.resize(length);
	workspace.exp_value.resize(length);
	for (uint64_t step = steps; step-- > 0;) {
		tensor_exp_with_level_index_(logs + step * log_stride, workspace.exp_value.data(),
			log_degree, boundary_degree, layout.offset.data(),
			workspace.powers, workspace.temp);
		sig_combine_backprop_with_level_index_(boundary.data() + step * length,
			workspace.exp_value.data(), workspace.d_current.data(),
			workspace.d_previous.data(), workspace.d_exp.data(), boundary_degree,
			layout.offset.data());
		for (uint64_t i = 0; i < length; ++i)
			workspace.d_previous[i] += coarse_adjoint[step * length + i];
		tensor_exp_backprop_with_level_index_(d_logs + step * log_stride,
			workspace.d_exp.data(), logs + step * log_stride, log_degree, boundary_degree,
			layout.offset.data(), workspace.powers, workspace.d_powers);
		workspace.d_current.swap(workspace.d_previous);
	}
}

template<std::floating_point T>
void solve_backward(
	const Layout& layout,
	const T* logs_x,
	const T* logs_y,
	T* d_logs_x,
	T* d_logs_y,
	const T* derivs,
	const T* k_grid,
	uint64_t steps_x,
	uint64_t steps_y,
	uint64_t factor_x,
	uint64_t factor_y,
	bool return_grid,
	Workspace<T>& workspace
) {
	const uint64_t nodes_x = steps_x * factor_x + 1;
	const uint64_t nodes_y = steps_y * factor_y + 1;
	workspace.prepare(layout, steps_x, steps_y, nodes_x, nodes_y, true);
	prepare_forward_data(layout, logs_x, logs_y, steps_x, steps_y, factor_x, factor_y, workspace);
	build_tape(layout, steps_x, steps_y, factor_x, factor_y, k_grid, workspace);
	std::fill(workspace.adjoint.begin(), workspace.adjoint.end(), static_cast<T>(0));
	std::fill(d_logs_x, d_logs_x + steps_x * layout.x_len, static_cast<T>(0));
	std::fill(d_logs_y, d_logs_y + steps_y * layout.y_len, static_cast<T>(0));
	if (return_grid) {
		for (uint64_t node = 0; node < nodes_x * nodes_y; ++node)
			workspace.adjoint[node * layout.state_len] = derivs[node];
	}
	else {
		workspace.adjoint[((nodes_x * nodes_y) - 1) * layout.state_len] = derivs[0];
	}

	for (uint64_t i = nodes_x; i-- > 1;) {
		const uint64_t coarse_x = (i - 1) / factor_x;
		const T* dx = workspace.scaled_x.data() + coarse_x * layout.x_len;
		for (uint64_t j = nodes_y; j-- > 1;) {
			const uint64_t coarse_y = (j - 1) / factor_y;
			const T* dy = workspace.scaled_y.data() + coarse_y * layout.y_len;
			cell_backward(layout, dx, dy,
				workspace.tape.data() + ((i - 1) * nodes_y + j - 1) * layout.state_len,
				workspace.tape.data() + ((i - 1) * nodes_y + j) * layout.state_len,
				workspace.tape.data() + (i * nodes_y + j - 1) * layout.state_len,
				workspace.tape.data() + (i * nodes_y + j) * layout.state_len,
				workspace.cell_cache.data() +
				((i - 1) * (nodes_y - 1) + j - 1) * layout.cell_cache_len,
				workspace.adjoint.data() + (i * nodes_y + j) * layout.state_len,
				workspace.adjoint.data() + ((i - 1) * nodes_y + j - 1) * layout.state_len,
				workspace.adjoint.data() + ((i - 1) * nodes_y + j) * layout.state_len,
				workspace.adjoint.data() + (i * nodes_y + j - 1) * layout.state_len,
				workspace.cell);
			for (uint64_t k = 0; k < layout.x_len; ++k)
				d_logs_x[coarse_x * layout.x_len + k] += workspace.cell.d_dx[k] / static_cast<T>(factor_x);
			for (uint64_t k = 0; k < layout.y_len; ++k)
				d_logs_y[coarse_y * layout.y_len + k] += workspace.cell.d_dy[k] / static_cast<T>(factor_y);
		}
	}

	if (layout.f_len != 0) {
		workspace.coarse_adj.assign((steps_x + 1) * layout.f_len, static_cast<T>(0));
		for (uint64_t i = 0; i < nodes_x; ++i) {
			const uint64_t coarse = std::min(i / factor_x, steps_x);
			const T* grad = workspace.adjoint.data() + i * nodes_y * layout.state_len + 1;
			T* coarse_grad = workspace.coarse_adj.data() + coarse * layout.f_len;
			for (uint64_t k = 0; k < layout.f_len; ++k) coarse_grad[k] += grad[k];
		}
		boundary_backward(layout, logs_x, steps_x, layout.degree_x, layout.degree_f,
			workspace.boundary_x, workspace.coarse_adj, d_logs_x, workspace);
	}
	if (layout.g_len != 0) {
		workspace.coarse_adj.assign((steps_y + 1) * layout.g_len, static_cast<T>(0));
		for (uint64_t j = 0; j < nodes_y; ++j) {
			const uint64_t coarse = std::min(j / factor_y, steps_y);
			const T* grad = workspace.adjoint.data() + j * layout.state_len + 1 + layout.f_len;
			T* coarse_grad = workspace.coarse_adj.data() + coarse * layout.g_len;
			for (uint64_t k = 0; k < layout.g_len; ++k) coarse_grad[k] += grad[k];
		}
		boundary_backward(layout, logs_y, steps_y, layout.degree_y, layout.degree_g,
			workspace.boundary_y, workspace.coarse_adj, d_logs_y, workspace);
	}
}

}  // namespace log_pde_detail

template<std::floating_point T>
void sig_kernel_log_pde_(
	const T* path_x,
	const T* path_y,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length_x,
	uint64_t length_y,
	uint64_t log_step_x,
	uint64_t log_step_y,
	uint64_t degree_x,
	uint64_t degree_y,
	uint64_t dyadic_order_x,
	uint64_t dyadic_order_y,
	bool return_grid,
	int n_jobs
) {
	using namespace log_pde_detail;
	if (length_x <= 1 || length_y <= 1)
		throw std::invalid_argument("log-PDE requires paths with at least two points");
	if (log_step_x == 0 || log_step_y == 0)
		throw std::invalid_argument("log-PDE block sizes must be positive");
	const uint64_t intervals_x = length_x - 1;
	const uint64_t intervals_y = length_y - 1;
	if (intervals_x % log_step_x != 0 || intervals_y % log_step_y != 0)
		throw std::invalid_argument("log-PDE block size must divide the path intervals");
	const uint64_t steps_x = intervals_x / log_step_x;
	const uint64_t steps_y = intervals_y / log_step_y;
	const Layout layout(dimension, degree_x, degree_y);
	const uint64_t factor_x = refinement(dyadic_order_x);
	const uint64_t factor_y = refinement(dyadic_order_y);
	const uint64_t nodes_x = fine_steps(steps_x, factor_x) + 1;
	const uint64_t nodes_y = fine_steps(steps_y, factor_y) + 1;
	const uint64_t out_stride = return_grid ? nodes_x * nodes_y : 1;
	const uint64_t path_stride_x = length_x * dimension;
	const uint64_t path_stride_y = length_y * dimension;

	auto worker = [&](uint64_t start, uint64_t end) {
		Workspace<T> workspace;
		std::vector<T> logs_x(steps_x * layout.x_len);
		std::vector<T> logs_y(steps_y * layout.y_len);
		std::vector<T> signature;
		for (uint64_t batch = start; batch < end; ++batch) {
			build_path_logs(
				path_x + batch * path_stride_x, logs_x.data(), dimension,
				steps_x, log_step_x, degree_x, signature
			);
			build_path_logs(
				path_y + batch * path_stride_y, logs_y.data(), dimension,
				steps_y, log_step_y, degree_y, signature
			);
			solve_forward(
				layout, logs_x.data(), logs_y.data(), out + batch * out_stride,
				steps_x, steps_y, factor_x, factor_y, return_grid, workspace
			);
		}
	};
	if (batch_size == 0 || n_jobs == 1 || batch_size == 1) worker(0, batch_size);
	else spawn_batch_threads(batch_size, n_jobs, worker);
}

template<std::floating_point T>
void sig_kernel_log_pde_backprop_(
	const T* path_x,
	const T* path_y,
	T* d_path_x,
	T* d_path_y,
	const T* derivs,
	const T* k_grid,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length_x,
	uint64_t length_y,
	uint64_t log_step_x,
	uint64_t log_step_y,
	uint64_t degree_x,
	uint64_t degree_y,
	uint64_t dyadic_order_x,
	uint64_t dyadic_order_y,
	bool return_grid,
	int n_jobs
) {
	using namespace log_pde_detail;
	if (length_x <= 1 || length_y <= 1)
		throw std::invalid_argument("log-PDE requires paths with at least two points");
	if (log_step_x == 0 || log_step_y == 0)
		throw std::invalid_argument("log-PDE block sizes must be positive");
	const uint64_t intervals_x = length_x - 1;
	const uint64_t intervals_y = length_y - 1;
	if (intervals_x % log_step_x != 0 || intervals_y % log_step_y != 0)
		throw std::invalid_argument("log-PDE block size must divide the path intervals");
	const uint64_t steps_x = intervals_x / log_step_x;
	const uint64_t steps_y = intervals_y / log_step_y;
	const Layout layout(dimension, degree_x, degree_y);
	const uint64_t factor_x = refinement(dyadic_order_x);
	const uint64_t factor_y = refinement(dyadic_order_y);
	const uint64_t nodes_x = fine_steps(steps_x, factor_x) + 1;
	const uint64_t nodes_y = fine_steps(steps_y, factor_y) + 1;
	const uint64_t deriv_stride = return_grid ? nodes_x * nodes_y : 1;
	const uint64_t grid_stride = nodes_x * nodes_y;
	const uint64_t path_stride_x = length_x * dimension;
	const uint64_t path_stride_y = length_y * dimension;

	auto worker = [&](uint64_t start, uint64_t end) {
		Workspace<T> workspace;
		std::vector<T> logs_x(steps_x * layout.x_len);
		std::vector<T> logs_y(steps_y * layout.y_len);
		std::vector<T> d_logs_x(steps_x * layout.x_len);
		std::vector<T> d_logs_y(steps_y * layout.y_len);
		std::vector<T> signature;
		std::vector<T> log_derivs;
		std::vector<T> sig_derivs;
		std::vector<T> block_derivs;
		for (uint64_t batch = start; batch < end; ++batch) {
			const T* batch_path_x = path_x + batch * path_stride_x;
			const T* batch_path_y = path_y + batch * path_stride_y;
			build_path_logs(
				batch_path_x, logs_x.data(), dimension, steps_x,
				log_step_x, degree_x, signature
			);
			build_path_logs(
				batch_path_y, logs_y.data(), dimension, steps_y,
				log_step_y, degree_y, signature
			);
			solve_backward(
				layout, logs_x.data(), logs_y.data(), d_logs_x.data(), d_logs_y.data(),
				derivs + batch * deriv_stride,
				k_grid == nullptr ? nullptr : k_grid + batch * grid_stride,
				steps_x, steps_y, factor_x, factor_y, return_grid, workspace
			);
			path_logs_backward(
				batch_path_x, d_logs_x.data(), d_path_x + batch * path_stride_x,
				dimension, length_x, steps_x, log_step_x, degree_x,
				signature, log_derivs, sig_derivs, block_derivs
			);
			path_logs_backward(
				batch_path_y, d_logs_y.data(), d_path_y + batch * path_stride_y,
				dimension, length_y, steps_y, log_step_y, degree_y,
				signature, log_derivs, sig_derivs, block_derivs
			);
		}
	};
	if (batch_size == 0 || n_jobs == 1 || batch_size == 1) worker(0, batch_size);
	else spawn_batch_threads(batch_size, n_jobs, worker);
}
