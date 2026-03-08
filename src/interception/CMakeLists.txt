cmake_minimum_required(VERSION 3.14)
project(gtp_proxy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_executable(gtp_proxy gtp_proxy.cpp)
target_link_libraries(gtp_proxy PRIVATE nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(gtp_proxy PRIVATE /W4 /utf-8)
endif()
