find_package(CUDAToolkit REQUIRED)

add_executable(test_app
    dll_funcs.cpp
    test_app.cpp
    tests.cpp
    utils.cpp
)

target_compile_features(test_app PRIVATE cxx_std_20)
target_include_directories(test_app PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(test_app PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})

if(TARGET cpsig)
    add_dependencies(test_app cpsig)
    add_custom_command(TARGET test_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:cpsig>
            $<TARGET_FILE_DIR:test_app>
    )
    if(WIN32)
        add_custom_command(TARGET test_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:tbb>
                $<TARGET_FILE_DIR:test_app>
        )
    endif()
endif()

if(TARGET cusig)
    add_dependencies(test_app cusig)
    add_custom_command(TARGET test_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:cusig>
            $<TARGET_FILE_DIR:test_app>
    )
endif()
