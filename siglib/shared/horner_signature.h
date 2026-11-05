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

// Included inside the backend's target namespace with HORNER_INLINE and HORNER_LARGE defined.
template<int Dimension, int Degree>
struct HornerSignature {
	template<int Level> static constexpr int width = [] {
		int result = 1;
		for (int i = 0; i < Level; ++i) result *= Dimension;
		return result;
	}();
	static constexpr int size = (width<Degree + 1> - Dimension) / (Dimension - 1);
	static constexpr int dimension = Dimension;
	static constexpr int degree = Degree;
	static constexpr int state_size = size;
	static constexpr int signature_size = size;
	template<int Level> static constexpr int offset = (width<Level> - Dimension) / (Dimension - 1);

	template<int Target, int Level, typename T, size_t... Index>
	HORNER_INLINE static void stage(T* state, const T* increment, T* work, std::index_sequence<Index...>) {
		constexpr int count = width<Level>;
		constexpr int offset = (count - Dimension) / (Dimension - 1);
		if constexpr (Target == 1) {
			((state[Index] = state[Index] + increment[Index]), ...);
		}
		else if constexpr (Level == 1) {
			((work[Index] = state[Index] + T(1.0 / Target) * increment[Index]), ...);
		}
		else if constexpr (Level == Target) {
			((state[offset + Index] = state[offset + Index] + work[Index / Dimension] * increment[Index % Dimension]), ...);
		}
		else {
			// Descending writes preserve the previous level's prefixes in work.
			((work[count - 1 - Index] = state[offset + count - 1 - Index]
				+ T(1.0 / (Target - Level + 1)) * (work[(count - 1 - Index) / Dimension]
					* increment[(count - 1 - Index) % Dimension])), ...);
		}
		if constexpr (Level < Target)
			stage<Target, Level + 1>(state, increment, work, std::make_index_sequence<width<Level + 1>>{});
	}

	template<int Target = Degree, typename T>
	HORNER_INLINE static void update_levels(T* state, const T* increment, T* work) {
		stage<Target, 1>(state, increment, work, std::make_index_sequence<Dimension>{});
		if constexpr (Target > 1) update_levels<Target - 1>(state, increment, work);
	}

	template<typename T>
	HORNER_LARGE static void update_large(T* state, const T* increment) {
		T work[width<Degree - 1>];
		update_levels(state, increment, work);
	}

	template<typename T>
	HORNER_INLINE static void update(T* state, const T* increment) {
		if constexpr (size > 120) update_large(state, increment);
		else {
			T work[width<Degree - 1>];
			update_levels(state, increment, work);
		}
	}

	template<int Target, int Level = 1, typename T, size_t... Index>
	HORNER_INLINE static void intermediates(const T* state, const T* increment, T* work, std::index_sequence<Index...>) {
		if constexpr (Level == 1)
			((work[Index] = state[Index] + T(1.0 / Target) * increment[Index]), ...);
		else
			((work[offset<Level> + Index] = state[offset<Level> + Index]
				+ T(1.0 / (Target - Level + 1)) * (work[offset<Level - 1> + Index / Dimension] * increment[Index % Dimension])), ...);
		if constexpr (Level + 1 < Target)
			intermediates<Target, Level + 1>(state, increment, work, std::make_index_sequence<width<Level + 1>>{});
	}

	template<int Target, int Level, size_t Index, typename T>
	HORNER_INLINE static void reverse_word(const T* increment, const T* work, const T* incoming, T* gradient, T* ds, T* dv) {
		const T value = incoming[Index];
		if constexpr (Level < Target) ds[offset<Level> + Index] += value;
		const T scaled = T(1.0 / (Target - Level + 1)) * value;
		dv[Index % Dimension] += scaled * work[offset<Level - 1> + Index / Dimension];
		if constexpr (Index % Dimension == 0) gradient[Index / Dimension] = scaled * increment[Index % Dimension];
		else gradient[Index / Dimension] += scaled * increment[Index % Dimension];
	}

	template<int Target, int Level, typename T, size_t... Index>
	HORNER_INLINE static void reverse_stage(const T* increment, const T* work, const T* incoming, T* gradient, T* ds, T* dv, std::index_sequence<Index...>) {
		if constexpr (Level == 1) {
			((ds[Index] += incoming[Index], dv[Index] += T(1.0 / Target) * incoming[Index]), ...);
		}
		else {
			// Ascending writes preserve unread children in gradient.
			(reverse_word<Target, Level, Index>(increment, work, incoming, gradient, ds, dv), ...);
			reverse_stage<Target, Level - 1>(increment, work, gradient, gradient, ds, dv, std::make_index_sequence<width<Level - 1>>{});
		}
	}

	template<int Target = Degree, typename T>
	HORNER_INLINE static void pullback_levels(const T* state, const T* increment, const T* g, T* ds, T* dv) {
		T work[offset<Target>];
		T gradient[width<Target - 1>];
		intermediates<Target>(state, increment, work, std::make_index_sequence<Dimension>{});
		reverse_stage<Target, Target>(increment, work, g + offset<Target>, gradient, ds, dv, std::make_index_sequence<width<Target>>{});
		if constexpr (Target > 2) pullback_levels<Target - 1>(state, increment, g, ds, dv);
	}

	template<typename T>
	HORNER_INLINE static void pullback_impl(const T* state, const T* increment, const T* g, T* ds, T* dv) {
		T gradient[offset<Degree>];
		for (int i = 0; i < offset<Degree>; ++i) gradient[i] = g[i];
		for (int i = 0; i < Dimension; ++i) dv[i] = g[i];
		pullback_levels(state, increment, g, gradient, dv);
		for (int i = 0; i < offset<Degree>; ++i) ds[i] = gradient[i];
		if (ds != g) for (int i = offset<Degree>; i < size; ++i) ds[i] = g[i];
	}

	template<typename T>
	HORNER_LARGE static void pullback_large(const T* state, const T* increment, const T* g, T* ds, T* dv) {
		pullback_impl(state, increment, g, ds, dv);
	}

	template<typename T>
	HORNER_INLINE static void pullback(const T* state, const T* increment, const T* g, T* ds, T* dv) {
		if constexpr (size > 84 || Degree >= 5) pullback_large(state, increment, g, ds, dv);
		else pullback_impl(state, increment, g, ds, dv);
	}

	template<typename T>
	HORNER_INLINE static void expand(const T* state, T* out) {
		for (int i = 0; i < size; ++i) out[i] = state[i];
	}

	template<typename T>
	HORNER_INLINE static void expand_pullback(const T*, const T* g, T* ds) { expand(g, ds); }
};

template<typename Run>
bool dispatch_horner(uint64_t dimension, uint64_t degree, Run run) {
	auto by_dimension = [&]<int dim>() {
		switch (degree) {
		case 2: return run.template operator()<HornerSignature<dim, 2>>();
		case 3: return run.template operator()<HornerSignature<dim, 3>>();
		case 4: if constexpr (dim <= 4) return run.template operator()<HornerSignature<dim, 4>>(); break;
		case 5: if constexpr (dim <= 3) return run.template operator()<HornerSignature<dim, 5>>(); break;
		case 6: if constexpr (dim == 2) return run.template operator()<HornerSignature<dim, 6>>(); break;
		}
		return false;
	};
	switch (dimension) {
	case 2: return by_dimension.template operator()<2>();
	case 3: return by_dimension.template operator()<3>();
	case 4: return by_dimension.template operator()<4>();
	case 5: return by_dimension.template operator()<5>();
	case 6: return by_dimension.template operator()<6>();
	case 7: return by_dimension.template operator()<7>();
	case 8: return by_dimension.template operator()<8>();
	}
	return false;
}
