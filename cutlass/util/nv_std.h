/******************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * \brief C++ features that may be otherwise unimplemented for CUDA device functions.
 *
 * This file has three components:
 *
 *   (1) Macros:
 *       - Empty macro defines for C++ keywords not supported by the current
 *         version of C++. These simply allow compilation to proceed (but do
 *         not provide the added semantics).
 *           - \p noexcept
 *           - \p constexpr
 *           - \p nullptr
 *           - \p static_assert
 *
 *       - Macro functions that we need in constant expressions because the
 *         C++ equivalents require constexpr compiler support.  These are
 *         prefixed with \p __NV_STD_*
 *           - \p __NV_STD_MAX
 *           - \p __NV_STD_MIN
 *
 *   (2) Re-implementations of STL functions and types:
 *       - C++ features that need the \p __device__ annotation.  These are
 *         placed into the \p nv_std namespace.
 *           - \p plus
 *           - \p less
 *           - \p greater
 *           - \p min
 *           - \p max
 *           - \p methods on std::pair (==, !=, <, <=, >, >=, and make_pair())
 *
 *   (3) Stop-gap implementations of unsupported STL functions and types:
 *       - STL functions and types defined by C++ 11/14/17/etc. that are not
 *         provided by the current version of C++. These are placed into the
 *         \p nv_std namespace
 *           - \p integral_constant
 *           - \p nullptr_t
 *           - \p true_type
 *           - \p false_type
 *           - \p bool_constant
 *           - \p enable_if
 *           - \p conditional
 *           - \p is_same
 *           - \p is_base_of
 *           - \p remove_const
 *           - \p remove_volatile
 *           - \p remove_cv
 *           - \p is_volatile
 *           - \p is_pointer
 *           - \p is_void
 *           - \p is_integral
 *           - \p is_floating_point
 *           - \p is_arithmetic
 *           - \p is_fundamental
 *           - \p is_trivially_copyable
 *           - \p alignment_of
 *           - \p aligned_storage
 *
 *   (4) Functions and types that are STL-like (but aren't in the STL):
 *           - \p TODO: min and max functors?
 *
 * The idea is that, as we drop support for older compilers, we can simply #define
 * the \p __NV_STD_XYZ macros and \p nv_std namespace to alias their C++
 * counterparts (or trivially find-and-replace their occurrences in code text).
 */


//-----------------------------------------------------------------------------
// Include STL files that nv_std provides functionality for
//-----------------------------------------------------------------------------

#include <cstddef>          // nullptr_t
#include <algorithm>        // Minimum/maximum operations
#include <functional>       // Arithmetic operations
#include <utility>          // For methods on std::pair
#if (!defined(_MSC_VER) && (__cplusplus >= 201103L)) || (defined(_MSC_VER) && (_MS_VER >= 1500))
    #include <type_traits>  // For integral constants, conditional metaprogramming, and type traits
#endif



/******************************************************************************
 * Macros
 ******************************************************************************/
//-----------------------------------------------------------------------------
// Keywords
//-----------------------------------------------------------------------------

/// noexcept, constexpr
#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1900))
    #ifndef noexcept
        #define noexcept
    #endif
    #ifndef constexpr
        #define constexpr
    #endif
#endif

/// nullptr
#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1310 ))
    #ifndef nullptr
        #define nullptr 0
    #endif
#endif

/// static_assert
#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1600 ))
    #ifndef static_assert
        #define __nv_std_cat_(a, b) a ## b
        #define __nv_std_cat(a, b) __nv_std_cat_(a, b)
        #define static_assert(__e, __m) typedef int __nv_std_cat(AsSeRt, __LINE__)[(__e) ? 1 : -1]
    #endif
#endif


//-----------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------

/// Select maximum(a, b)
#ifndef __NV_STD_MAX
    #define __NV_STD_MAX(a, b) (((b) > (a)) ? (b) : (a))
#endif

/// Select minimum(a, b)
#ifndef __NV_STD_MIN
    #define __NV_STD_MIN(a, b) (((b) < (a)) ? (b) : (a))
#endif




/******************************************************************************
 * Re-implementations
 ******************************************************************************/

namespace nv_std {

    //-----------------------------------------------------------------------------
    // Arithmetic operations, comparisons <functional>
    //-----------------------------------------------------------------------------

    /// nv_std::plus
    template <typename T>
    struct plus
    {
        inline __host__ __device__
        constexpr T operator()(const T &lhs, const T &rhs) const
        {
            return lhs + rhs;
        }
    };


    /// std::less
    template <typename T>
    struct less
    {
        inline __host__ __device__
        constexpr bool operator()(const T &lhs, const T &rhs) const
        {
            return lhs < rhs;
        }
    };

    /// std::greater
    template <typename T>
    struct greater
    {
        inline __host__ __device__
        constexpr bool operator()(const T &lhs, const T &rhs) const
        {
            return lhs > rhs;
        }
    };


    //-----------------------------------------------------------------------------
    // Minimum/maximum operations <algorithm>
    //-----------------------------------------------------------------------------

    /// std::min
    template <typename T>
    inline __host__ __device__
    constexpr const T& min(
        const T& a,
        const T& b)
    {
        return (b < a) ? b : a;
    }

    /// std::max
    template <typename T>
    inline __host__ __device__
    constexpr const T& max(
        const T& a,
        const T& b)
    {
        return (a < b) ? b : a;
    }


    //-----------------------------------------------------------------------------
    // Methods on std::pair
    //-----------------------------------------------------------------------------

    using std::pair;

    template< class T1, class T2 >
    inline __host__ __device__
    constexpr bool operator==( const pair<T1,T2>& lhs, const pair<T1,T2>& rhs )
    {
        return (lhs.first == rhs.first) && (lhs.second == rhs.second);
    }

    template< class T1, class T2 >
    inline __host__ __device__
    constexpr bool operator!=( const pair<T1,T2>& lhs, const pair<T1,T2>& rhs )
    {
        return (lhs.first != rhs.first) && (lhs.second != rhs.second);
    }

    template< class T1, class T2 >
    inline __host__ __device__
    constexpr bool operator<( const pair<T1,T2>& lhs, const pair<T1,T2>& rhs )
    {
        return (lhs.first < rhs.first) ?
            true :
            (rhs.first < lhs.first) ?
                false :
                (lhs.second < rhs.second);
    }

    template< class T1, class T2 >
    inline __host__ __device__
    constexpr bool operator<=( const pair<T1,T2>& lhs, const pair<T1,T2>& rhs )
    {
        return !(rhs < lhs);
    }

    template< class T1, class T2 >
    inline __host__ __device__
    constexpr bool operator>( const pair<T1,T2>& lhs, const pair<T1,T2>& rhs )
    {
        return (rhs < lhs);
    }

    template< class T1, class T2 >
    inline __host__ __device__
    constexpr bool operator>=( const pair<T1,T2>& lhs, const pair<T1,T2>& rhs )
    {
        return !(lhs < rhs);
    }

    template< class T1, class T2 >
    inline __host__ __device__
    std::pair<T1,T2> make_pair( T1 t, T2 u )
    {
        std::pair<T1,T2> retval;
        retval.first = t;
        retval.second = u;
        return retval;
    }

}   // namespace nv_std



/******************************************************************************
 * Implementations of C++ 11/14/17/... STL features
 ******************************************************************************/

namespace nv_std {

//-----------------------------------------------------------------------------
// Integral constant helper types <type_traits>
//-----------------------------------------------------------------------------

#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

    /// std::integral_constant
    template <typename value_t, value_t V>
    struct integral_constant;

    /// std::integral_constant
    template <typename value_t, value_t V>
    struct integral_constant
    {
        static const value_t value = V;

        typedef value_t                           value_type;
        typedef integral_constant<value_t, V>     type;

        inline __host__ __device__ operator value_type() const
        {
             return value;
        }

        inline __host__ __device__ const value_type operator()() const
        {
            return value;
        }
    };


#else

    using std::integral_constant;
    using std::pair;

#endif

    /// The type used as a compile-time boolean with true value.
    typedef integral_constant<bool, true>   true_type;

    /// The type used as a compile-time boolean with false value.
    typedef integral_constant<bool, false>  false_type;

    // TEMPORARY LAGUNAS
//#if (!defined(_MSC_VER) && (__cplusplus < 201402L)) || (defined(_MSC_VER) && (_MSC_VER < 1900))

    /// std::bool_constant
    template <bool V>
    struct bool_constant : nv_std::integral_constant<bool, V>
    {};

//#else

//    using std::bool_constant;

//#endif


#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1700))

    /// std::nullptr_t
    struct nullptr_t {};

#else

    using std::nullptr_t;

#endif



    //-----------------------------------------------------------------------------
    // Conditional metaprogramming <type_traits>
    //-----------------------------------------------------------------------------

#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1600))

    /// std::enable_if (true specialization)
    template<bool C, typename T = void>
    struct enable_if {
      typedef T type;
    };

    /// std::enable_if (false specialization)
    template<typename T>
    struct enable_if<false, T> { };


    /// std::conditional (true specialization)
    template<bool B, class T, class F>
    struct conditional { typedef T type; };

    /// std::conditional (false specialization)
    template<class T, class F>
    struct conditional<false, T, F> { typedef F type; };

#else

    using std::enable_if;
    using std::conditional;

#endif



    //-----------------------------------------------------------------------------
    // Const/volatility specifiers <type_traits>
    //-----------------------------------------------------------------------------

#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

    /// std::remove_const (non-const specialization)
    template <typename T> struct remove_const                   { typedef T type; };

    /// std::remove_const (const specialization)
    template <typename T> struct remove_const<const T>          { typedef T type; };



    /// std::remove_volatile (non-volatile specialization)
    template <typename T> struct remove_volatile                { typedef T type; };

    /// std::remove_volatile (volatile specialization)
    template <typename T> struct remove_volatile<volatile T>    { typedef T type; };



    /// std::remove_cv
    template <typename T>
    struct remove_cv {
        typedef typename remove_volatile<typename remove_const<T>::type>::type type;
    };

#else

    using std::remove_const;
    using std::remove_volatile;
    using std::remove_cv;

#endif


    //-----------------------------------------------------------------------------
    // Type relationships <type_traits>
    //-----------------------------------------------------------------------------

#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

    /// std::is_same (false specialization)
    template <typename A, typename B>
    struct is_same : false_type
    {};

    /// std::is_same (true specialization)
    template <typename A>
    struct is_same<A, A> : true_type
    {};


    /// Helper for std::is_base_of
    template<typename BaseT, typename DerivedT>
    struct is_base_of_helper
    {
        typedef char (&yes)[1];
        typedef char (&no)[2];

        template<typename B, typename D>
        struct dummy
        {
            operator B*() const;
            operator D*();
        };

        template<typename T>
        static yes check(DerivedT*, T);

        static no check(BaseT*, int);

        static const bool value = sizeof(check(dummy<BaseT, DerivedT>(), int())) == sizeof(yes);
    };

    /// std::is_base_of
    template <typename BaseT, typename DerivedT>
    struct is_base_of : integral_constant<
        bool,
        (is_base_of_helper<typename remove_cv<BaseT>::type, typename remove_cv<DerivedT>::type>::value) ||
            (is_same<typename remove_cv<BaseT>::type, typename remove_cv<DerivedT>::type>::value)>
    {};


#else

    using std::is_same;
    using std::is_base_of;

#endif



    //-----------------------------------------------------------------------------
    // Type properties <type_traits>
    //-----------------------------------------------------------------------------

#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))

    /// std::is_volatile
    template <typename T> struct is_volatile             : false_type {};
    template <typename T> struct is_volatile<volatile T> : true_type {};


    /// Helper for std::is_pointer (false specialization)
    template <typename T> struct is_pointer_helper        : false_type {};

    /// Helper for std::is_pointer (true specialization)
    template <typename T> struct is_pointer_helper<T*>    : true_type {};

    /// std::is_pointer
    template <typename T> struct is_pointer : is_pointer_helper<typename remove_cv<T>::type> {};



    /// std::is_void
    template <typename T>
    struct is_void : is_same<void, typename remove_cv<T>::type>
    {};



    /// std::is_integral
    template <typename T> struct is_integral            : false_type {};
    template <> struct is_integral<char>                : true_type {};
    template <> struct is_integral<signed char>         : true_type {};
    template <> struct is_integral<unsigned char>       : true_type {};
    template <> struct is_integral<short>               : true_type {};
    template <> struct is_integral<unsigned short>      : true_type {};
    template <> struct is_integral<int>                 : true_type {};
    template <> struct is_integral<unsigned int>        : true_type {};
    template <> struct is_integral<long>                : true_type {};
    template <> struct is_integral<unsigned long>       : true_type {};
    template <> struct is_integral<long long>           : true_type {};
    template <> struct is_integral<unsigned long long>  : true_type {};
    template <typename T> struct is_integral<volatile T>        : is_integral<T> {};
    template <typename T> struct is_integral<const T>           : is_integral<T> {};
    template <typename T> struct is_integral<const volatile T>  : is_integral<T> {};



    /// std::is_floating_point
    template <typename T>
    struct is_floating_point : integral_constant<
        bool,
        (is_same<float, typename remove_cv<T>::type>::value ||
            is_same<double, typename remove_cv<T>::type>::value)>
    {};



    /// std::is_arithmetic
    template <typename T>
    struct is_arithmetic :
        integral_constant<bool, (is_integral<T>::value || is_floating_point<T>::value)>
    {};


    /// std::is_fundamental
    template <typename T>
    struct is_fundamental : integral_constant<
        bool, (is_arithmetic<T>::value ||
            is_void<T>::value ||
            is_same<nullptr_t, typename remove_cv<T>::type>::value)>
    {};




#else

    using std::is_volatile;
    using std::is_pointer;
    using std::is_void;
    using std::is_integral;
    using std::is_floating_point;
    using std::is_arithmetic;
    using std::is_fundamental;

#endif


#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) ||      \
    (defined(_MSC_VER) && (_MSC_VER < 1800)) ||             \
    (defined(__GNUG__) && (__GNUC__ < 5))

    /**
     * std::is_trivially_copyable
     *
     * This implementation only evaluates true if T is fundamental or pointer
     *
     * Without help from partial template specializations provided by the user for
     * a specific class or struct, this trait will never report that the specified
     * class or struct  is trivially-copyable ; this is always safe,
     * if possibly sub-optimal.
     */
    template <typename T>
    struct is_trivially_copyable :
        integral_constant<bool, (is_fundamental<T>::value || is_pointer<T>::value)>
    {};

#else

    using std::is_trivially_copyable;

#endif




    //-----------------------------------------------------------------------------
    // Alignment and layout utilities
    //-----------------------------------------------------------------------------

#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1500))


    /// std::alignment_of
    template <typename value_t>
    struct alignment_of
    {
        struct pad
        {
            value_t val;
            char    byte;
        };

        enum
        {
            value = sizeof(pad) - sizeof(value_t)
        };
    };

#else

    template <typename value_t>
    struct alignment_of : std::alignment_of<value_t> {};

#endif

    /* 16B specializations where 32-bit Win32 host compiler disagrees with device compiler */
    template <> struct alignment_of<int4>                 { enum { value = 16 }; };
    template <> struct alignment_of<uint4>                { enum { value = 16 }; };
    template <> struct alignment_of<float4>               { enum { value = 16 }; };
    template <> struct alignment_of<long4>                { enum { value = 16 }; };
    template <> struct alignment_of<ulong4>               { enum { value = 16 }; };
    template <> struct alignment_of<longlong2>            { enum { value = 16 }; };
    template <> struct alignment_of<ulonglong2>           { enum { value = 16 }; };
    template <> struct alignment_of<double2>              { enum { value = 16 }; };
    template <> struct alignment_of<longlong4>            { enum { value = 16 }; };
    template <> struct alignment_of<ulonglong4>           { enum { value = 16 }; };
    template <> struct alignment_of<double4>              { enum { value = 16 }; };

    // Specializations for volatile/const qualified types
    template <typename value_t> struct alignment_of<volatile value_t>       : alignment_of<value_t> {};
    template <typename value_t> struct alignment_of<const value_t>          : alignment_of<value_t> {};
    template <typename value_t> struct alignment_of<const volatile value_t> : alignment_of<value_t> {};



#if (!defined(_MSC_VER) && (__cplusplus < 201103L)) || (defined(_MSC_VER) && (_MSC_VER < 1800))

    template<size_t Align> struct aligned_chunk;
    template<> struct __align__(1) aligned_chunk<1> { uint8_t buff; };
    template<> struct __align__(2) aligned_chunk<2> { uint16_t buff; };
    template<> struct __align__(4) aligned_chunk<4> { uint32_t buff; };
    template<> struct __align__(8) aligned_chunk<8> { uint32_t buff[2]; };
    template<> struct __align__(16) aligned_chunk<16> { uint32_t buff[4]; };
    template<> struct __align__(32) aligned_chunk<32> { uint32_t buff[8]; };
    template<> struct __align__(64) aligned_chunk<64> { uint32_t buff[16]; };
    template<> struct __align__(128) aligned_chunk<128> { uint32_t buff[32]; };
    template<> struct __align__(256) aligned_chunk<256> { uint32_t buff[64]; };
    template<> struct __align__(512) aligned_chunk<512> { uint32_t buff[128]; };
    template<> struct __align__(1024) aligned_chunk<1024> { uint32_t buff[256]; };
    template<> struct __align__(2048) aligned_chunk<2048> { uint32_t buff[512]; };
    template<> struct __align__(4096) aligned_chunk<4096> { uint32_t buff[1024]; };

    /// std::aligned_storage
    template <size_t Len, size_t Align>
    struct aligned_storage
    {
        typedef aligned_chunk<Align> type[Len / sizeof(aligned_chunk<Align>)];
    };

#else

    using std::aligned_storage;

#endif




}; // namespace nv_std

