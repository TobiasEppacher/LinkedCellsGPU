#pragma once

#include <Kokkos_Core.hpp>

template <typename T, typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
class KokkosVector {
public:
    using value_type = T;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    using size_type = typename memory_space::size_type;

    KokkosVector() : _data("kokkos_vector_data", 0) {}

    explicit KokkosVector(size_type n) : _data("kokkos_vector_data", n) {}

    void clear() { _size = 0; }

    size_type size() const { return _size; }

    size_type capacity() const { return _data.extent(0); }

    void resize(size_type n) { Kokkos::resize(_data, n); }

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_type i) const { return _data(i); }

    void push_back(const T& value) {
        if (_size == _data.extent(0)) {
            resize(_size == 0 ? 8 : _data.extent(0) * 2);
        }
        _data(_size++) = value;
    }

    Kokkos::View<T*, memory_space> active_view() {
        return Kokkos::subview(_data, Kokkos::make_pair(size_type(0), _size));
    }

private:
    Kokkos::View<T*, memory_space> _data;
    size_type _size = 0;
};