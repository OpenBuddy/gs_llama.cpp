set(TARGET gs)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(${TARGET} gs.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
# add include path websocketpp
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()
