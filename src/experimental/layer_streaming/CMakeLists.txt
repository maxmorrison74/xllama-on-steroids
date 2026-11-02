# Opt-in research target. Deliberately not linked into xllama or the UWP app.
add_library(xllama-layer-streaming STATIC layer_streaming.cpp expert_cache.cpp moe_streaming.cpp moe_file.cpp)
target_compile_features(xllama-layer-streaming PUBLIC cxx_std_17)
target_include_directories(xllama-layer-streaming PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(xllama-layer-streaming PRIVATE
    ${PROJECT_SOURCE_DIR}/llama.cpp/src)
target_link_libraries(xllama-layer-streaming PUBLIC llama)
if(XLLAMA_ENABLE_ASAN)
    target_compile_options(xllama-layer-streaming PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(xllama-layer-streaming PUBLIC -fsanitize=address)
endif()
if(XLLAMA_ENABLE_UBSAN)
    target_compile_options(xllama-layer-streaming PUBLIC -fsanitize=undefined)
    target_link_options(xllama-layer-streaming PUBLIC -fsanitize=undefined)
endif()

add_executable(xllama-streaming-spike main.cpp)
target_link_libraries(xllama-streaming-spike PRIVATE xllama-layer-streaming)

add_executable(xllama-streaming-smoke smoke.cpp)
target_link_libraries(xllama-streaming-smoke PRIVATE xllama-layer-streaming)
add_test(NAME streaming-smoke COMMAND xllama-streaming-smoke)

add_executable(xllama-moe-streaming-smoke moe_smoke.cpp)
target_link_libraries(xllama-moe-streaming-smoke PRIVATE xllama-layer-streaming)
add_test(NAME moe-streaming-smoke COMMAND xllama-moe-streaming-smoke)

add_executable(xllama-moe-streaming-spike moe_main.cpp)
target_link_libraries(xllama-moe-streaming-spike PRIVATE xllama-layer-streaming)
