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

#include "cu_sig_kernel_log_pde_tensor.cuh"

namespace log_pde_cuda_detail {

template<typename T>
struct CellScratch {
	T gamma;
	T u_base;
	T u_provisional;
	View<T> dx_adj_dy;
	View<T> dy_adj_dx;
	View<T> df0;
	View<T> df1;
	View<T> fp;
	View<T> dg0;
	View<T> dg1;
	View<T> gp;
	View<T> ad_fse;
	View<T> ad_gse;
	View<T> ad_df0;
	View<T> ad_df1;
	View<T> ad_fp;
	View<T> ad_dg0;
	View<T> ad_dg1;
	View<T> ad_gp;
	View<T> ad_dx_adj_dy;
	View<T> ad_dy_adj_dx;
	View<T> d_dx;
	View<T> d_dy;
};

template<typename T>
__device__ CellScratch<T> make_forward_cell_scratch(View<T> storage, const Params& p) {
	uint64_t cursor = 0;
	CellScratch<T> scratch{};
	auto take = [&](uint64_t length) {
		View<T> result = storage.sub(cursor);
		cursor += length;
		return result;
	};
	scratch.dx_adj_dy = take(p.f_len);
	scratch.dy_adj_dx = take(p.g_len);
	scratch.df0 = take(p.f_len);
	scratch.df1 = take(p.f_len);
	scratch.fp = take(p.f_len);
	scratch.dg0 = take(p.g_len);
	scratch.dg1 = take(p.g_len);
	scratch.gp = take(p.g_len);
	return scratch;
}

template<typename T>
__device__ CellScratch<T> make_backward_cell_scratch(View<T> storage, const Params& p) {
	uint64_t cursor = 0;
	CellScratch<T> scratch{};
	auto take = [&](uint64_t length) {
		View<T> result = storage.sub(cursor);
		cursor += length;
		return result;
	};
	scratch.ad_fse = take(p.f_len);
	scratch.ad_gse = take(p.g_len);
	scratch.ad_df0 = take(p.f_len);
	scratch.ad_df1 = take(p.f_len);
	scratch.ad_fp = scratch.ad_fse;
	scratch.ad_dg0 = take(p.g_len);
	scratch.ad_dg1 = take(p.g_len);
	scratch.ad_gp = scratch.ad_gse;
	scratch.ad_dx_adj_dy = take(p.f_len);
	scratch.ad_dy_adj_dx = take(p.g_len);
	scratch.d_dx = take(p.x_len);
	scratch.d_dy = take(p.y_len);
	return scratch;
}

template<typename T>
__device__ void increment(
	const Params& p, ConstView<T> drive, uint64_t drive_degree, T u,
	ConstView<T> state, uint64_t state_degree,
	ConstView<T> other_state, uint64_t other_degree, View<T> out
) {
	const uint64_t length = state_degree == 0 ? 0 : p.offset[state_degree + 1];
	for (uint64_t i = 0; i < length; ++i) out[i] = static_cast<T>(0);
	const uint64_t product_degree = drive_degree < state_degree ?
		drive_degree : state_degree;
	const uint64_t drive_length = product_degree == 0 ? 0 : p.offset[product_degree + 1];
	for (uint64_t i = 0; i < drive_length; ++i) out[i] += u * drive[i];
	tensor_product_add(p, state, state_degree, drive, product_degree,
		out, state_degree);
	tensor_adjoint_add(p, other_state, other_degree, drive, drive_degree,
		out, state_degree, true);
}

template<typename T>
__device__ void increment_backward(
	const Params& p, ConstView<T> drive, uint64_t drive_degree, T u,
	ConstView<T> state, uint64_t state_degree,
	ConstView<T> other_state, uint64_t other_degree, ConstView<T> d_out,
	T& d_u, View<T> d_state, View<T> d_other_state, View<T> d_drive
) {
	const uint64_t product_degree = drive_degree < state_degree ?
		drive_degree : state_degree;
	const uint64_t drive_length = product_degree == 0 ? 0 : p.offset[product_degree + 1];
	T d_u_increment = static_cast<T>(0);
	for (uint64_t i = 0; i < drive_length; ++i) d_u_increment += drive[i] * d_out[i];
	d_u += d_u_increment;
	for (uint64_t i = 0; i < drive_length; ++i) d_drive[i] += u * d_out[i];
	tensor_product_backward(p, state, state_degree, drive, product_degree,
		d_out, state_degree, d_state, d_drive);
	tensor_adjoint_backward(p, other_state, other_degree, drive, drive_degree,
		d_out, state_degree, d_other_state, d_drive, true);
}

template<typename T>
__device__ T forcing(
	const Params& p, T u, ConstView<T> f, ConstView<T> g, T gamma,
	ConstView<T> dx_adj_dy, ConstView<T> dy_adj_dx
) {
	T result = u * gamma;
	for (uint64_t i = 0; i < p.f_len; ++i) result += f[i] * dx_adj_dy[i];
	for (uint64_t i = 0; i < p.g_len; ++i) result += g[i] * dy_adj_dx[i];
	return result;
}

template<typename T>
__device__ void forcing_backward(
	const Params& p, T d_out, T u, ConstView<T> f, ConstView<T> g, T gamma,
	ConstView<T> dx_adj_dy, ConstView<T> dy_adj_dx,
	T& d_u, View<T> d_f, View<T> d_g, T& d_gamma,
	View<T> d_dx_adj_dy, View<T> d_dy_adj_dx
) {
	d_u += d_out * gamma;
	d_gamma += d_out * u;
	for (uint64_t i = 0; i < p.f_len; ++i) {
		d_f[i] += d_out * dx_adj_dy[i];
		d_dx_adj_dy[i] += d_out * f[i];
	}
	for (uint64_t i = 0; i < p.g_len; ++i) {
		d_g[i] += d_out * dy_adj_dx[i];
		d_dy_adj_dx[i] += d_out * g[i];
	}
}

template<typename T>
__device__ void cell_forward(
	const Params& p, ConstView<T> dx, ConstView<T> dy,
	ConstView<T> nw, ConstView<T> north, ConstView<T> west, View<T> se,
	bool has_known_u, T known_u, CellScratch<T>& scratch
) {
	const ConstView<T> f_nw = nw.sub(1);
	const ConstView<T> f_n = north.sub(1);
	const ConstView<T> f_w = west.sub(1);
	const ConstView<T> g_nw = f_nw.sub(p.f_len);
	const ConstView<T> g_n = f_n.sub(p.f_len);
	const ConstView<T> g_w = f_w.sub(p.f_len);
	const View<T> f_se = se.sub(1);
	const View<T> g_se = f_se.sub(p.f_len);

	for (uint64_t i = 0; i < p.f_len; ++i)
		scratch.dx_adj_dy[i] = static_cast<T>(0);
	for (uint64_t i = 0; i < p.g_len; ++i)
		scratch.dy_adj_dx[i] = static_cast<T>(0);
	tensor_adjoint_add(p, dx, p.degree_x, dy, p.degree_y,
		scratch.dx_adj_dy, p.degree_f, false);
	tensor_adjoint_add(p, dy, p.degree_y, dx, p.degree_x,
		scratch.dy_adj_dx, p.degree_g, false);
	const uint64_t common_degree = p.degree_x < p.degree_y ? p.degree_x : p.degree_y;
	const uint64_t common_length = common_degree == 0 ? 0 : p.offset[common_degree + 1];
	scratch.gamma = static_cast<T>(0);
	for (uint64_t i = 0; i < common_length; ++i) scratch.gamma += dx[i] * dy[i];

	increment(p, dx, p.degree_x, north[0], f_n, p.degree_f,
		g_n, p.degree_g, scratch.df0);
	increment(p, dy, p.degree_y, west[0], g_w, p.degree_g,
		f_w, p.degree_f, scratch.dg0);
	for (uint64_t i = 0; i < p.f_len; ++i) scratch.fp[i] = f_n[i] + scratch.df0[i];
	for (uint64_t i = 0; i < p.g_len; ++i) scratch.gp[i] = g_w[i] + scratch.dg0[i];

	const T f_nw_value = forcing(p, nw[0], f_nw, g_nw, scratch.gamma,
		scratch.dx_adj_dy.read(), scratch.dy_adj_dx.read());
	scratch.u_base = north[0] + west[0] - nw[0];
	scratch.u_provisional = scratch.u_base + f_nw_value;

	increment(p, dx, p.degree_x, scratch.u_provisional,
		scratch.fp.read(), p.degree_f, scratch.gp.read(), p.degree_g, scratch.df1);
	increment(p, dy, p.degree_y, scratch.u_provisional,
		scratch.gp.read(), p.degree_g, scratch.fp.read(), p.degree_f, scratch.dg1);
	for (uint64_t i = 0; i < p.f_len; ++i)
		f_se[i] = f_n[i] + static_cast<T>(0.5) * (scratch.df0[i] + scratch.df1[i]);
	for (uint64_t i = 0; i < p.g_len; ++i)
		g_se[i] = g_w[i] + static_cast<T>(0.5) * (scratch.dg0[i] + scratch.dg1[i]);

	if (has_known_u) {
		se[0] = known_u;
		return;
	}
	const T f_n_value = forcing(p, north[0], f_n, g_n, scratch.gamma,
		scratch.dx_adj_dy.read(), scratch.dy_adj_dx.read());
	const T f_w_value = forcing(p, west[0], f_w, g_w, scratch.gamma,
		scratch.dx_adj_dy.read(), scratch.dy_adj_dx.read());
	const T f_corner = forcing(p, scratch.u_provisional,
		f_se.read(), g_se.read(), scratch.gamma,
		scratch.dx_adj_dy.read(), scratch.dy_adj_dx.read());
	se[0] = scratch.u_base + static_cast<T>(0.25) *
		(f_nw_value + f_n_value + f_w_value + f_corner);
}

template<typename T>
__device__ void cell_backward(
	const Params& p, ConstView<T> dx, ConstView<T> dy,
	ConstView<T> nw, ConstView<T> north, ConstView<T> west, ConstView<T> se,
	ConstView<T> cache, ConstView<T> d_se,
	View<T> d_nw, View<T> d_north, View<T> d_west, CellScratch<T>& scratch
) {
	const ConstView<T> f_nw = nw.sub(1);
	const ConstView<T> f_n = north.sub(1);
	const ConstView<T> f_w = west.sub(1);
	const ConstView<T> g_nw = f_nw.sub(p.f_len);
	const ConstView<T> g_n = f_n.sub(p.f_len);
	const ConstView<T> g_w = f_w.sub(p.f_len);
	const ConstView<T> f_se = se.sub(1);
	const ConstView<T> g_se = f_se.sub(p.f_len);
	const ConstView<T> dx_adj_dy = cache;
	const ConstView<T> dy_adj_dx = dx_adj_dy.sub(p.f_len);
	const ConstView<T> fp = dy_adj_dx.sub(p.g_len);
	const ConstView<T> gp = fp.sub(p.f_len);
	const ConstView<T> scalars = gp.sub(p.g_len);
	const View<T> d_f_nw = d_nw.sub(1);
	const View<T> d_f_n = d_north.sub(1);
	const View<T> d_f_w = d_west.sub(1);
	const View<T> d_g_nw = d_f_nw.sub(p.f_len);
	const View<T> d_g_n = d_f_n.sub(p.f_len);
	const View<T> d_g_w = d_f_w.sub(p.f_len);

	for (uint64_t i = 0; i < p.f_len; ++i) {
		scratch.ad_fse[i] = d_se[1 + i];
		scratch.ad_df0[i] = static_cast<T>(0);
		scratch.ad_df1[i] = static_cast<T>(0);
		scratch.ad_dx_adj_dy[i] = static_cast<T>(0);
	}
	for (uint64_t i = 0; i < p.g_len; ++i) {
		scratch.ad_gse[i] = d_se[1 + p.f_len + i];
		scratch.ad_dg0[i] = static_cast<T>(0);
		scratch.ad_dg1[i] = static_cast<T>(0);
		scratch.ad_dy_adj_dx[i] = static_cast<T>(0);
	}
	for (uint64_t i = 0; i < p.x_len; ++i) scratch.d_dx[i] = static_cast<T>(0);
	for (uint64_t i = 0; i < p.y_len; ++i) scratch.d_dy[i] = static_cast<T>(0);

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

	forcing_backward(p, d_f_corner, u_provisional, f_se, g_se, gamma,
		dx_adj_dy, dy_adj_dx, d_u_provisional, scratch.ad_fse, scratch.ad_gse,
		d_gamma, scratch.ad_dx_adj_dy, scratch.ad_dy_adj_dx);

	for (uint64_t i = 0; i < p.f_len; ++i) {
		d_f_n[i] += scratch.ad_fse[i];
		scratch.ad_df0[i] += static_cast<T>(0.5) * scratch.ad_fse[i];
		scratch.ad_df1[i] += static_cast<T>(0.5) * scratch.ad_fse[i];
	}
	for (uint64_t i = 0; i < p.g_len; ++i) {
		d_g_w[i] += scratch.ad_gse[i];
		scratch.ad_dg0[i] += static_cast<T>(0.5) * scratch.ad_gse[i];
		scratch.ad_dg1[i] += static_cast<T>(0.5) * scratch.ad_gse[i];
	}
	for (uint64_t i = 0; i < p.f_len; ++i) scratch.ad_fp[i] = static_cast<T>(0);
	for (uint64_t i = 0; i < p.g_len; ++i) scratch.ad_gp[i] = static_cast<T>(0);

	increment_backward(p, dx, p.degree_x, u_provisional,
		fp, p.degree_f, gp, p.degree_g, scratch.ad_df1.read(),
		d_u_provisional, scratch.ad_fp, scratch.ad_gp, scratch.d_dx);
	increment_backward(p, dy, p.degree_y, u_provisional,
		gp, p.degree_g, fp, p.degree_f, scratch.ad_dg1.read(),
		d_u_provisional, scratch.ad_gp, scratch.ad_fp, scratch.d_dy);

	for (uint64_t i = 0; i < p.f_len; ++i) {
		d_f_n[i] += scratch.ad_fp[i];
		scratch.ad_df0[i] += scratch.ad_fp[i];
	}
	for (uint64_t i = 0; i < p.g_len; ++i) {
		d_g_w[i] += scratch.ad_gp[i];
		scratch.ad_dg0[i] += scratch.ad_gp[i];
	}

	d_u_base += d_u_provisional;
	d_f_nw_value += d_u_provisional;
	forcing_backward(p, d_f_nw_value, nw[0], f_nw, g_nw, gamma,
		dx_adj_dy, dy_adj_dx, d_nw[0], d_f_nw, d_g_nw, d_gamma,
		scratch.ad_dx_adj_dy, scratch.ad_dy_adj_dx);
	forcing_backward(p, d_f_n_value, north[0], f_n, g_n, gamma,
		dx_adj_dy, dy_adj_dx, d_north[0], d_f_n, d_g_n, d_gamma,
		scratch.ad_dx_adj_dy, scratch.ad_dy_adj_dx);
	forcing_backward(p, d_f_w_value, west[0], f_w, g_w, gamma,
		dx_adj_dy, dy_adj_dx, d_west[0], d_f_w, d_g_w, d_gamma,
		scratch.ad_dx_adj_dy, scratch.ad_dy_adj_dx);

	increment_backward(p, dx, p.degree_x, north[0],
		f_n, p.degree_f, g_n, p.degree_g, scratch.ad_df0.read(),
		d_north[0], d_f_n, d_g_n, scratch.d_dx);
	increment_backward(p, dy, p.degree_y, west[0],
		g_w, p.degree_g, f_w, p.degree_f, scratch.ad_dg0.read(),
		d_west[0], d_g_w, d_f_w, scratch.d_dy);

	d_north[0] += d_u_base;
	d_west[0] += d_u_base;
	d_nw[0] -= d_u_base;
	const uint64_t common_degree = p.degree_x < p.degree_y ? p.degree_x : p.degree_y;
	const uint64_t common_length = common_degree == 0 ? 0 : p.offset[common_degree + 1];
	for (uint64_t i = 0; i < common_length; ++i) {
		scratch.d_dx[i] += d_gamma * dy[i];
		scratch.d_dy[i] += d_gamma * dx[i];
	}
	tensor_adjoint_backward(p, dx, p.degree_x, dy, p.degree_y,
		scratch.ad_dx_adj_dy.read(), p.degree_f, scratch.d_dx, scratch.d_dy, false);
	tensor_adjoint_backward(p, dy, p.degree_y, dx, p.degree_x,
		scratch.ad_dy_adj_dx.read(), p.degree_g, scratch.d_dy, scratch.d_dx, false);
}

template<typename T>
__device__ void build_boundary(
	const Params& p, ConstView<T> logs, uint64_t steps,
	uint64_t log_degree, uint64_t boundary_degree, View<T> boundary, View<T> scratch
) {
	const uint64_t length = boundary_degree == 0 ? 0 : p.offset[boundary_degree + 1];
	if (length == 0) return;
	for (uint64_t i = 0; i < (steps + 1) * length; ++i)
		boundary[i] = static_cast<T>(0);
	const View<T> exp_value = scratch;
	const View<T> boundary_next = exp_value.sub(length);
	const View<T> powers = boundary_next.sub(length);
	const View<T> temp = powers.sub(length);
	const uint64_t log_stride = log_degree == 0 ? 0 : p.offset[log_degree + 1];
	for (uint64_t step = 0; step < steps; ++step) {
		tensor_exp(p, logs.sub(step * log_stride), exp_value, log_degree,
			boundary_degree, powers, temp);
		sig_combine(p, boundary.sub(step * length).read(), exp_value.read(),
			boundary_next, boundary_degree);
		const View<T> destination = boundary.sub((step + 1) * length);
		for (uint64_t i = 0; i < length; ++i) destination[i] = boundary_next[i];
	}
}

template<typename T>
__device__ void boundary_backward(
	const Params& p, ConstView<T> logs, uint64_t steps, uint64_t nodes,
	uint64_t shift, uint64_t log_degree, uint64_t boundary_degree,
	ConstView<T> boundary, ConstView<T> adjoint,
	uint64_t node_stride, uint64_t state_offset, View<T> coarse_adjoint,
	View<T> d_logs, View<T> scratch
) {
	const uint64_t length = boundary_degree == 0 ? 0 : p.offset[boundary_degree + 1];
	if (length == 0) return;
	for (uint64_t i = 0; i < (steps + 1) * length; ++i)
		coarse_adjoint[i] = static_cast<T>(0);
	for (uint64_t node = 0; node < nodes; ++node) {
		const uint64_t shifted = node >> shift;
		const uint64_t coarse = shifted < steps ? shifted : steps;
		const ConstView<T> grad = adjoint.sub(node * node_stride + state_offset);
		const View<T> destination = coarse_adjoint.sub(coarse * length);
		for (uint64_t i = 0; i < length; ++i) destination[i] += grad[i];
	}
	const uint64_t log_stride = log_degree == 0 ? 0 : p.offset[log_degree + 1];
	View<T> d_current = scratch;
	View<T> d_previous = d_current.sub(length);
	const View<T> d_exp = d_previous.sub(length);
	const View<T> exp_value = d_exp.sub(length);
	const View<T> powers = exp_value.sub(length);
	const View<T> d_powers = powers.sub(boundary_degree * length);
	for (uint64_t i = 0; i < length; ++i)
		d_current[i] = coarse_adjoint[(steps * length) + i];
	for (uint64_t step = steps; step-- > 0;) {
		tensor_exp(p, logs.sub(step * log_stride), exp_value, log_degree,
			boundary_degree, powers, d_powers);
		sig_combine_backward(p, boundary.sub(step * length), exp_value.read(),
			d_current.read(), d_previous, d_exp, boundary_degree);
		for (uint64_t i = 0; i < length; ++i)
			d_previous[i] += coarse_adjoint[step * length + i];
		tensor_exp_backward(p, d_logs.sub(step * log_stride), d_exp.read(),
			logs.sub(step * log_stride), log_degree, boundary_degree, powers, d_powers);
		const View<T> old_current = d_current;
		d_current = d_previous;
		d_previous = old_current;
	}
}

}  // namespace log_pde_cuda_detail
