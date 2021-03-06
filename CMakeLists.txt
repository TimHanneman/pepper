CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

# name of the project is PEPPER
PROJECT(pepper)
set (CMAKE_CXX_FLAGS "-fPIC -O3 -pipe")
set (CMAKE_C_FLAGS "-fPIC -O3 -pipe -shared -rdynamic")

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

FIND_PACKAGE(PythonInterp 3 REQUIRED)
FIND_PACKAGE(PythonLibs 3 REQUIRED)

if (PYTHONINTERP_FOUND)
    message("Python found")
else()
    message("Python not found")
endif()

# enable installing dependencies
option(INSTALL_DEPENDENCIES
        "Install project dependencies"
        ON)

INCLUDE(pepper/modules/htslib.cmake)
INCLUDE(pepper/modules/pybind11.cmake)

# pybind11 to interface
pybind11_add_module(PEPPER pepper/modules/src/pybind_api.cpp)

add_dependencies(PEPPER htslib)

# add all the external libraries
target_link_libraries(PEPPER PRIVATE z)
target_link_libraries(PEPPER PRIVATE bz2)
target_link_libraries(PEPPER PRIVATE curl)
target_link_libraries(PEPPER PRIVATE lzma)
target_link_libraries(PEPPER PRIVATE ${HTSLIB_SRC_DIR}/libhts.a)

# PEPPER_SNP
pybind11_add_module(PEPPER_SNP pepper_snp/modules/src/pybind_api.cpp)

add_dependencies(PEPPER_SNP htslib)

# add all the external libraries
target_link_libraries(PEPPER_SNP PRIVATE z)
target_link_libraries(PEPPER_SNP PRIVATE bz2)
target_link_libraries(PEPPER_SNP PRIVATE curl)
target_link_libraries(PEPPER_SNP PRIVATE lzma)
target_link_libraries(PEPPER_SNP PRIVATE ${HTSLIB_SRC_DIR}/libhts.a)

# PEPPER_HP
pybind11_add_module(PEPPER_HP pepper_hp/modules/src/pybind_api.cpp)

add_dependencies(PEPPER_HP htslib)

# add all the external libraries
target_link_libraries(PEPPER_HP PRIVATE z)
target_link_libraries(PEPPER_HP PRIVATE bz2)
target_link_libraries(PEPPER_HP PRIVATE curl)
target_link_libraries(PEPPER_HP PRIVATE lzma)
target_link_libraries(PEPPER_HP PRIVATE ${HTSLIB_SRC_DIR}/libhts.a)

#-------------------
# Test
#-------------------
#ENABLE_TESTING()
#ADD_SUBDIRECTORY(modules/unit_tests)

