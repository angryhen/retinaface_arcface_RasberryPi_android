# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("testopencv")

# opencv
set(libs "${CMAKE_SOURCE_DIR}/src/main/jniLibs")
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/include)
add_library(libopencv_java4 SHARED IMPORTED )
set_target_properties(libopencv_java4 PROPERTIES IMPORTED_LOCATION "${libs}/${ANDROID_ABI}/libopencv_java4.so")

# ncnn
# import ncnn library
add_library(ncnn STATIC IMPORTED)
# change this folder path to yours
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${libs}/ncnn-android-vulkan-lib/${ANDROID_ABI}/libncnn.a)
include_directories(${libs}/ncnn-android-vulkan-lib/include/ncnn)

add_library(glslang STATIC IMPORTED)
add_library(OGLCompiler STATIC IMPORTED)
add_library(OSDependent STATIC IMPORTED)
add_library(SPIRV STATIC IMPORTED)
set_target_properties(glslang PROPERTIES IMPORTED_LOCATION ${libs}/ncnn-android-vulkan-lib/${ANDROID_ABI}/libglslang.a)
set_target_properties(OGLCompiler PROPERTIES IMPORTED_LOCATION ${libs}/ncnn-android-vulkan-lib/${ANDROID_ABI}/libOGLCompiler.a)
set_target_properties(OSDependent PROPERTIES IMPORTED_LOCATION ${libs}/ncnn-android-vulkan-lib/${ANDROID_ABI}/libOSDependent.a)
set_target_properties(SPIRV PROPERTIES IMPORTED_LOCATION ${libs}/ncnn-android-vulkan-lib/${ANDROID_ABI}/libSPIRV.a)

# openmp
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp")

if(DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")

# disable rtti and exceptions
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")


file(GLOB SRC src/main/cpp/*.cpp src/main/cpp/*.h src/main/cpp/*.c src/main/cpp/*.hpp)
set(SRC_CODE ${SRC})
add_library(native-lib SHARED ${SRC_CODE})

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries(native-lib
        ncnn
        glslang SPIRV OGLCompiler OSDependent
        android
        z
        log
        jnigraphics
        vulkan
        libopencv_java4
        )
