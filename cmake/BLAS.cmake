# Copyright 2026 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

include_guard(GLOBAL)

include(ExternalProject)

set(_blas_root "${CMAKE_BINARY_DIR}/pysiglib-openblas")
set(_blas_install "${_blas_root}/install")
file(MAKE_DIRECTORY "${_blas_install}/include/openblas")
set(_blas_cmake "${CMAKE_COMMAND}")
set(_blas_compiler_args)
if(MSVC)
    find_program(_blas_clang clang-cl HINTS "$ENV{ProgramFiles}/LLVM/bin" REQUIRED)
    find_program(_blas_vswhere vswhere HINTS "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/Installer" REQUIRED)
    execute_process(COMMAND "${_blas_vswhere}" -latest -products * -property installationPath
        OUTPUT_VARIABLE _blas_vs OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    file(WRITE "${_blas_root}/cmake-msvc.cmd"
        "@call \"${_blas_vs}/VC/Auxiliary/Build/vcvars64.bat\" >nul\r\n@\"${CMAKE_COMMAND}\" %*\r\n")
    set(_blas_cmake "${_blas_root}/cmake-msvc.cmd")
    list(APPEND _blas_compiler_args "-DCMAKE_C_COMPILER=${_blas_clang}" "-DMSVC_STATIC_CRT=ON"
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded" "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW"
        "-DCMAKE_C_FLAGS=/clang:-Wno-unsafe-buffer-usage")
    set(_blas_library "${_blas_install}/lib/openblas.lib")
else()
    enable_language(C)
    list(APPEND _blas_compiler_args "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
    set(_blas_library "${_blas_install}/lib/libopenblas.a")
endif()
find_program(_blas_ninja ninja REQUIRED)
ExternalProject_Add(pysiglib_openblas_build
    PREFIX "${_blas_root}"
    URL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34.tar.gz
    URL_HASH SHA256=cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CONFIGURE_COMMAND "${_blas_cmake}" -S <SOURCE_DIR> -B <BINARY_DIR> -G Ninja
        "-DCMAKE_MAKE_PROGRAM=${_blas_ninja}" ${_blas_compiler_args}
        -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=${_blas_install}"
        -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DFIXED_LIBNAME=ON
        -DBUILD_WITHOUT_LAPACK=ON -DNOFORTRAN=ON -DBUILD_TESTING=OFF -DONLY_CBLAS=ON
        -DBUILD_SINGLE=ON -DBUILD_DOUBLE=ON -DBUILD_COMPLEX=OFF -DBUILD_COMPLEX16=OFF
        -DUSE_THREAD=OFF -DUSE_LOCKING=ON -DNUM_THREADS=256 -DDYNAMIC_ARCH=ON
    BUILD_COMMAND "${_blas_cmake}" --build <BINARY_DIR> --parallel 4
    INSTALL_COMMAND "${_blas_cmake}" --install <BINARY_DIR>
    BUILD_BYPRODUCTS "${_blas_library}"
)
add_library(pysiglib_openblas STATIC IMPORTED GLOBAL)
set_target_properties(pysiglib_openblas PROPERTIES IMPORTED_LOCATION "${_blas_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_blas_install}/include/openblas")
add_dependencies(pysiglib_openblas pysiglib_openblas_build)
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(pysiglib_openblas INTERFACE Threads::Threads m ${CMAKE_DL_LIBS})
    if(NOT APPLE)
        target_link_options(pysiglib_openblas INTERFACE "LINKER:--exclude-libs,libopenblas.a")
    endif()
endif()
if(NOT CPSIG_SKIP_INSTALL)
    ExternalProject_Get_Property(pysiglib_openblas_build SOURCE_DIR)
    install(FILES "${SOURCE_DIR}/LICENSE" DESTINATION pysiglib/licenses RENAME OpenBLAS-LICENSE.txt)
endif()
