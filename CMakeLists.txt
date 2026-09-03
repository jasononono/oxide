cmake_minimum_required(VERSION 4.1.1)

project(Oxide LANGUAGES CXX)

# language standards and build options

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RELEASE "build release library instead of test cpp script" ON)

# all .cpp src files

set(SOURCE
    src/backend.cpp
    src/common.cpp
    src/dispatcher.cpp
    src/functions.cpp
    src/oxide.cpp
    src/tensor.cpp
    src/util.cpp
)

set(NAME main)


# if RELEASE, use nanobind to produce the darwin.so library
if (RELEASE)


    set(NAME core)
    
    find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

    # find nanobind package from pip3 installation
    execute_process(COMMAND python3 -m nanobind --cmake_dir OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(${NAME} 
        ${SOURCE}
        src/binding.cpp
    )

    target_link_libraries(${NAME} PRIVATE
        Python::Module
    )


else()


    add_executable(main
        ${SOURCE}
        tests/main.cpp
    )


endif()


# resource files

configure_file(
    src/shader.metal
    ${CMAKE_BINARY_DIR}/shader.metal
    COPYONLY
)

# -I and -L links

target_include_directories(${NAME} PRIVATE
    ~/Documents/lib/metal-cpp
    src/include
    src/specialize
)

target_link_libraries(${NAME} PRIVATE
    "-framework Metal"
    "-framework Foundation"
)