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

template<std::floating_point T>
class PointImpl;

template<std::floating_point T>
class PointImplTimeAug;

template<std::floating_point T>
class PointImplLeadLag;

template<std::floating_point T>
class PointImplTimeAugLeadLag;

template<std::floating_point T>
class Point;

template<std::floating_point T>
class Path {
	static T compute_time_step(uint64_t length, T end_time) {
		return length > 1 ? end_time / (length - 1) : static_cast<T>(0);
	}
public:

	Path(const T* data_, uint64_t dimension_, uint64_t length_, bool time_aug_ = false, bool lead_lag_ = false, T end_time_ = 1.) :
		_dimension{ (lead_lag_ ? 2 * dimension_ : dimension_) + (time_aug_ ? 1 : 0) },
		_length{ lead_lag_ ? length_ * 2 - 1 : length_ },
		_data{ std::span<const T>(data_, dimension_ * length_) },
		_data_dimension{ dimension_ },
		_data_length{ length_ },
		_data_size{ dimension_ * length_ },
		_time_aug{ time_aug_ },
		_lead_lag{ lead_lag_ },
		_time_step{ compute_time_step(_length, end_time_) } {
	}

	Path(const std::span<const T> data_, uint64_t dimension_, uint64_t length_, bool time_aug_ = false, bool lead_lag_ = false, T end_time_ = 1.) :
		_dimension{ (lead_lag_ ? 2 * dimension_ : dimension_) + (time_aug_ ? 1 : 0) },
		_length{ lead_lag_ ? length_ * 2 - 1 : length_ },
		_data{ data_ },
		_data_dimension{ dimension_ },
		_data_length{ length_ },
		_data_size{ dimension_ * length_ },
		_time_aug{ time_aug_ },
		_lead_lag{ lead_lag_ },
		_time_step{ compute_time_step(_length, end_time_) } {
		if (data_.size() != dimension_ * length_)
			throw std::invalid_argument("1D vector is not the correct shape for a path of dimension " + std::to_string(dimension_) + " and length " + std::to_string(length_));
	}

	Path(const Path& other) :
		_dimension{ other._dimension },
		_length{ other._length },
		_data{ other._data },
		_data_dimension{ other._data_dimension },
		_data_length{ other._data_length },
		_data_size{ other._data_size },
		_time_aug{ other._time_aug },
		_lead_lag{ other._lead_lag },
		_time_step{ other._time_step } {
	}

	Path(const Path& other, bool time_aug_, bool lead_lag_, T end_time_ = 1.) :
		_dimension{ (lead_lag_ ? 2 * other._data_dimension : other._data_dimension) + (time_aug_ ? 1 : 0) },
		_length{ lead_lag_ ? other._data_length * 2 - 1 : other._data_length },
		_data{ other._data },
		_data_dimension{ other._data_dimension },
		_data_length{ other._data_length },
		_data_size{ other._data_size },
		_time_aug{ time_aug_ },
		_lead_lag{ lead_lag_ },
		_time_step{ compute_time_step(_length, end_time_) } {
	}

	~Path() = default;

	Path<T>& operator=(const Path&) = delete;

	inline uint64_t dimension() const { return _dimension; }
	inline uint64_t data_dimension() const { return _data_dimension; }
	inline uint64_t length() const { return _length; }
	inline uint64_t data_length() const { return _data_length; }
	inline const T* data() const { return _data.data(); }

	inline bool time_aug() const { return _time_aug; }
	inline bool lead_lag() const { return _lead_lag; }
	inline T time_step() const { return _time_step; }
	inline T end_time() const { return _time_step * (_length - 1); }

	friend class Point<T>;
	friend class PointImpl<T>;
	friend class PointImplTimeAug<T>;
	friend class PointImplLeadLag<T>;
	friend class PointImplTimeAugLeadLag<T>;

	Point<T> operator[](uint64_t i) const { 
#ifdef _DEBUG
		if (i >= _length)
			throw std::out_of_range("Argument out of bounds in Path::operator[]");
#endif
		return Point<T>(this, i);
	}

	inline Point<T> begin() const
	{
		return Point<T>(this, 0);
	}
	inline Point<T> end() const
	{
		return Point<T>(this, _length); 
	}

	bool operator==(const Path& other) const {
		return _data.data() == other._data.data()
			&& _time_aug == other._time_aug
			&& _lead_lag == other._lead_lag;
	}
	bool operator!=(const Path& other) const {
		return !this->operator==(other);
	}

private:
	const uint64_t _dimension;
	const uint64_t _length;

	const std::span<const T> _data;
	const uint64_t _data_dimension;
	const uint64_t _data_length;
	const uint64_t _data_size;

	const bool _time_aug;
	const bool _lead_lag;
	const T _time_step;
};

template<std::floating_point T>
class PointImpl {
	friend class Path<T>;
	friend class Point<T>;

public:
	PointImpl() : ptr{ nullptr }, path{ nullptr } {}
	PointImpl(const Path<T>* path_, uint64_t index) :
		ptr{ path_->_data.data() + index * path_->_data_dimension },
		path{ path_ }
	{}
	PointImpl(const PointImpl& other) = default;
	PointImpl& operator=(const PointImpl& other) = default;

	inline T operator[](uint64_t i) const { return ptr[i]; }
	inline void operator++() { ptr += path->_data_dimension; }
	inline void operator--() { ptr -= path->_data_dimension; }

	inline uint64_t dimension() const { return path->_dimension; }
	inline void advance(int64_t n) { ptr += n * path->_data_dimension; }
	inline void set_to_start() { ptr = path->_data.data(); }
	inline void set_to_end() { ptr = path->_data.data() + path->_data_size; }
	inline void set_to_index(int64_t n) { ptr = path->_data.data() + n * path->_data_dimension; }

	inline const T* data() const { return ptr; }
	inline uint64_t index() const { return static_cast<uint64_t>((ptr - path->_data.data()) / path->_data_dimension); }

#ifndef _DEBUG
	bool operator==(const PointImpl& other) const { return ptr == other.ptr; }
	bool operator<(const PointImpl& other) const { return ptr < other.ptr; }
	bool operator>(const PointImpl& other) const { return ptr > other.ptr; }
#else
	bool operator==(const PointImpl& other) const { return path == other.path && ptr == other.ptr; }
	bool operator<(const PointImpl& other) const { return path == other.path && ptr < other.ptr; }
	bool operator>(const PointImpl& other) const { return path == other.path && ptr > other.ptr; }
#endif
	bool operator!=(const PointImpl& other) const { return !(*this == other); }
	bool operator<=(const PointImpl& other) const { return !(*this > other); }
	bool operator>=(const PointImpl& other) const { return !(*this < other); }

	const T* ptr;
	const Path<T>* path;
};

template<std::floating_point T>
class PointImplTimeAug {
	friend class Path<T>;
	friend class Point<T>;

public:
	PointImplTimeAug() : ptr{ nullptr }, path{ nullptr }, time{ 0. } {}
	PointImplTimeAug(const Path<T>* path_, uint64_t index) :
		ptr{ path_->_data.data() + index * path_->_data_dimension },
		path{ path_ },
		time{ index * path_->_time_step }
	{}
	PointImplTimeAug(const PointImplTimeAug& other) = default;
	PointImplTimeAug& operator=(const PointImplTimeAug& other) = default;

	inline T operator[](uint64_t i) const { return (i < path->_data_dimension) ? ptr[i] : time; }
	inline void operator++() { ptr += path->_data_dimension; time += path->_time_step; }
	inline void operator--() { ptr -= path->_data_dimension; time -= path->_time_step; }

	inline uint64_t dimension() const { return path->_dimension; }
	inline void advance(int64_t n) { ptr += n * path->_data_dimension; time += n * path->_time_step; }
	inline void set_to_start() { ptr = path->_data.data(); time = 0.; }
	inline void set_to_end() { ptr = path->_data.data() + path->_data_size; time = path->_length * path->_time_step; }
	inline void set_to_index(int64_t n) { ptr = path->_data.data() + n * path->_data_dimension; time = n * path->_time_step; }

	inline const T* data() const { return ptr; }
	inline uint64_t index() const { return static_cast<uint64_t>((ptr - path->_data.data()) / path->_data_dimension); }

#ifndef _DEBUG
	bool operator==(const PointImplTimeAug& other) const { return ptr == other.ptr; }
	bool operator<(const PointImplTimeAug& other) const { return ptr < other.ptr; }
	bool operator>(const PointImplTimeAug& other) const { return ptr > other.ptr; }
#else
	bool operator==(const PointImplTimeAug& other) const { return path == other.path && ptr == other.ptr; }
	bool operator<(const PointImplTimeAug& other) const { return path == other.path && ptr < other.ptr; }
	bool operator>(const PointImplTimeAug& other) const { return path == other.path && ptr > other.ptr; }
#endif
	bool operator!=(const PointImplTimeAug& other) const { return !(*this == other); }
	bool operator<=(const PointImplTimeAug& other) const { return !(*this > other); }
	bool operator>=(const PointImplTimeAug& other) const { return !(*this < other); }

	const T* ptr;
	const Path<T>* path;

private:
	T time;
};

template<std::floating_point T>
class PointImplLeadLag {
	friend class Path<T>;
	friend class Point<T>;

public:
	PointImplLeadLag() : ptr{ nullptr }, path{ nullptr }, parity{ false } {}
	PointImplLeadLag(const Path<T>* path_, uint64_t index) :
		ptr{ path_->_data.data() + (index / 2) * path_->_data_dimension },
		path{ path_ },
		parity{ static_cast<bool>(index % 2) }
	{}
	PointImplLeadLag(const PointImplLeadLag& other) = default;
	PointImplLeadLag& operator=(const PointImplLeadLag& other) = default;

	inline T operator[](uint64_t i) const { 
		if (i < path->_data_dimension)
			return ptr[i];
		else {
			uint64_t leadIdx = parity ? i : i - path->_data_dimension;
			return ptr[leadIdx];
		}
	}
	inline void operator++() { if (parity) ptr += path->_data_dimension; parity = !parity; }
	inline void operator--() { if (!parity) ptr -= path->_data_dimension; parity = !parity; }

	inline uint64_t dimension() const { return path->_dimension; }
	inline void advance(int64_t n) { ptr += (n / 2) * path->_data_dimension; parity = (parity != static_cast<bool>(n % 2)); }
	inline void set_to_start() { ptr = path->_data.data(); parity = false; }
	inline void set_to_end() { ptr = path->_data.data() + path->_data_size; parity = true; }
	inline void set_to_index(int64_t n) { ptr = path->_data.data() + (n / 2) * path->_data_dimension; parity = static_cast<bool>(n % 2); }

	inline const T* data() const { return ptr; }
	inline uint64_t index() const { return 2 * static_cast<uint64_t>((ptr - path->_data.data()) / path->_data_dimension) + static_cast<uint64_t>(parity); }

#ifndef _DEBUG
	bool operator==(const PointImplLeadLag& other) const { return ptr == other.ptr && parity == other.parity; }
	bool operator<(const PointImplLeadLag& other) const { return ptr < other.ptr || (ptr == other.ptr && parity < other.parity); }
	bool operator>(const PointImplLeadLag& other) const { return ptr > other.ptr || (ptr == other.ptr && parity > other.parity); }
#else
	bool operator==(const PointImplLeadLag& other) const { return path == other.path && ptr == other.ptr && parity == other.parity; }
	bool operator<(const PointImplLeadLag& other) const { return path == other.path && (ptr < other.ptr || (ptr == other.ptr && parity < other.parity)); }
	bool operator>(const PointImplLeadLag& other) const { return path == other.path && (ptr > other.ptr || (ptr == other.ptr && parity > other.parity)); }
#endif
	bool operator!=(const PointImplLeadLag& other) const { return !(*this == other); }
	bool operator<=(const PointImplLeadLag& other) const { return !(*this > other); }
	bool operator>=(const PointImplLeadLag& other) const { return !(*this < other); }

	const T* ptr;
	const Path<T>* path;

private:
	bool parity;
};

template<std::floating_point T>
class PointImplTimeAugLeadLag {
	friend class Path<T>;
	friend class Point<T>;

public:
	PointImplTimeAugLeadLag() : ptr{ nullptr }, path{ nullptr }, parity{ false }, time{ 0. }, _data_dimension_times_2{ 0 } {}
	PointImplTimeAugLeadLag(const Path<T>* path_, uint64_t index) : 
		ptr{ path_->_data.data() + (index / 2) * path_->_data_dimension },
		path{ path_ },
		parity{ static_cast<bool>(index % 2) },
		time{ index * path_->_time_step },
		_data_dimension_times_2{ path_->_data_dimension * 2 }
	{}
	PointImplTimeAugLeadLag(const PointImplTimeAugLeadLag& other) = default;
	PointImplTimeAugLeadLag& operator=(const PointImplTimeAugLeadLag& other) = default;

	inline T operator[](uint64_t i) const {
		if (i < path->_data_dimension)
			return ptr[i];
		else if (i < _data_dimension_times_2) {
			uint64_t lead_idx = parity ? i : i - path->_data_dimension;
			return ptr[lead_idx];
		}
		else
			return time;
	}
	inline void operator++() {
		if (parity) { ptr += path->_data_dimension; }
		time += path->_time_step;
		parity = !parity;
	}
	inline void operator--() {
		if (!parity) { ptr -= path->_data_dimension; }
		time -= path->_time_step;
		parity = !parity;
	}

	inline uint64_t dimension() const { return path->_dimension; }
	inline void advance(int64_t n) { 
		ptr += (n / 2) * path->_data_dimension; 
		parity = (parity != static_cast<bool>(n % 2)); 
		time += n * path->_time_step;
	}
	inline void set_to_start() { ptr = path->_data.data(); parity = false; time = 0; }
	inline void set_to_end() { ptr = path->_data.data() + path->_data_size; parity = true; time = path->_length * path->_time_step; }
	inline void set_to_index(int64_t n) { 
		ptr = path->_data.data() + (n / 2) * path->_data_dimension;
		parity = static_cast<bool>(n % 2);
		time = n * path->_time_step;
	}

	inline const T* data() const { return ptr; }
	inline uint64_t index() const { return 2UL * static_cast<uint64_t>((ptr - path->_data.data()) / path->_data_dimension) + static_cast<uint64_t>(parity); }

#ifndef _DEBUG
	bool operator==(const PointImplTimeAugLeadLag& other) const { return ptr == other.ptr && parity == other.parity; }
	bool operator<(const PointImplTimeAugLeadLag& other) const { return ptr < other.ptr || (ptr == other.ptr && parity < other.parity); }
	bool operator>(const PointImplTimeAugLeadLag& other) const { return ptr > other.ptr || (ptr == other.ptr && parity > other.parity); }
#else
	bool operator==(const PointImplTimeAugLeadLag& other) const { return path == other.path && ptr == other.ptr && parity == other.parity; }
	bool operator<(const PointImplTimeAugLeadLag& other) const { return path == other.path && (ptr < other.ptr || (ptr == other.ptr && parity < other.parity)); }
	bool operator>(const PointImplTimeAugLeadLag& other) const { return path == other.path && (ptr > other.ptr || (ptr == other.ptr && parity > other.parity)); }
#endif
	bool operator!=(const PointImplTimeAugLeadLag& other) const { return !(*this == other); }
	bool operator<=(const PointImplTimeAugLeadLag& other) const { return !(*this > other); }
	bool operator>=(const PointImplTimeAugLeadLag& other) const { return !(*this < other); }

	const T* ptr;
	const Path<T>* path;

private:
	bool parity;
	T time;
	uint64_t _data_dimension_times_2;
};

template<std::floating_point T>
using PointVariant = std::variant<
	PointImpl<T>,
	PointImplTimeAug<T>,
	PointImplLeadLag<T>,
	PointImplTimeAugLeadLag<T>
>;

template<std::floating_point T>
class Point {
public:

	Point() : _impl{ PointImpl<T>{} } {}

	Point(const Path<T>* path, uint64_t index) {
		if (!path->_time_aug && !path->_lead_lag)
			_impl.template emplace<PointImpl<T>>(path, index);
		else if (path->_time_aug && !path->_lead_lag)
			_impl.template emplace<PointImplTimeAug<T>>(path, index);
		else if (!path->_time_aug && path->_lead_lag)
			_impl.template emplace<PointImplLeadLag<T>>(path, index);
		else
			_impl.template emplace<PointImplTimeAugLeadLag<T>>(path, index);
	}

	Point(const Point& other) = default;
	Point(Point&& other) noexcept = default;
	Point& operator=(const Point& other) = default;
	Point& operator=(Point&& other) noexcept = default;

	inline T operator[](uint64_t i) const { 
#ifdef _DEBUG
		sq_bracket_bounds_check(i);
#endif
		return std::visit([i](const auto& impl) -> T { return impl[i]; }, _impl);
	}
	inline Point& operator++() {
		std::visit([](auto& impl) { ++impl; }, _impl);
		return *this;
	}
	inline Point operator++(int) {
		Point tmp{ *this };
		++(*this);
		return tmp;
	}
	inline Point& operator--() {
		std::visit([](auto& impl) { --impl; }, _impl);
		return *this;
	}
	inline Point operator--(int) {
		Point tmp{ *this };
		--(*this);
		return tmp;
	}

	inline uint64_t dimension() const { return std::visit([](const auto& impl) { return impl.dimension(); }, _impl); }
	inline void advance(int64_t n) { 
		std::visit([n](auto& impl) { impl.advance(n); }, _impl);
	}
	inline void set_to_start() { std::visit([](auto& impl) { impl.set_to_start(); }, _impl); }
	inline void set_to_end() { std::visit([](auto& impl) { impl.set_to_end(); }, _impl); }
	inline void set_to_index(int64_t n) { std::visit([n](auto& impl) { impl.set_to_index(n); }, _impl); }

	inline const T* data() const { return std::visit([](const auto& impl) { return impl.data(); }, _impl); }
	inline uint64_t index() const { 
#ifdef _DEBUG
		index_bounds_check();
#endif
		return std::visit([](const auto& impl) { return impl.index(); }, _impl);
	}

	bool operator==(const Point& other) const {
		return std::visit([](const auto& a, const auto& b) -> bool {
			using A = std::decay_t<decltype(a)>;
			using B = std::decay_t<decltype(b)>;
			if constexpr (std::is_same_v<A, B>)
				return a == b;
			else
				return false; // Different impl types are never equal
		}, _impl, other._impl);
	}
	bool operator!=(const Point& other) const { return !(*this == other); }
	bool operator<(const Point& other) const {
		return std::visit([](const auto& a, const auto& b) -> bool {
			using A = std::decay_t<decltype(a)>;
			using B = std::decay_t<decltype(b)>;
			if constexpr (std::is_same_v<A, B>)
				return a < b;
			else
				return false;
		}, _impl, other._impl);
	}
	bool operator<=(const Point& other) const { return !(other < *this); }
	bool operator>(const Point& other) const { return other < *this; }
	bool operator>=(const Point& other) const { return !(*this < other); }

private:
	PointVariant<T> _impl;

#ifdef _DEBUG
	inline void sq_bracket_bounds_check(uint64_t i) const {
		std::visit([i](const auto& impl) {
			if (impl.ptr < impl.path->_data.data() || impl.ptr >= impl.path->_data.data() + impl.path->_data_size)
				throw std::out_of_range("Point is out of bounds for given path in Point::operator[]");
			if (i >= impl.path->_dimension)
				throw std::out_of_range("Argument out of bounds in Point::operator[]");
		}, _impl);
	}

	inline void index_bounds_check() const {
		std::visit([](const auto& impl) {
			if (impl.ptr < impl.path->_data.data() || impl.ptr >= impl.path->_data.data() + impl.path->_data_size)
				throw std::out_of_range("Point is out of bounds for given path in Point::index()");
		}, _impl);
	}
#endif
};
