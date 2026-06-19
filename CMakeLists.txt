cmake_minimum_required(VERSION 4.1.1)

project(Oxide LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main
    src/backend.cpp
    src/dispatcher.cpp
    src/functions.cpp
    src/oxide.cpp
    src/tensor.cpp
    src/typeutil.cpp
    
    main.cpp
)

configure_file(
    src/shader.metal
    ${CMAKE_BINARY_DIR}/shader.metal
    COPYONLY
)

target_include_directories(main PRIVATE
    external/metal-cpp
    external/nanobind/include
    src/include
)

target_link_libraries(main PRIVATE
    "-framework Metal"
    "-framework Foundation"
)