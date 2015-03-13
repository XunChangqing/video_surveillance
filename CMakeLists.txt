#############################################
# video_compression project
#############################################
cmake_minimum_required(VERSION 2.8)

project(video_compression)
set(VIDEO_SURVEILLANCE_VERSION_MAJOR 0)
set(VIDEO_SURVEILLANCE_VERSION_MINOR 1)

#set(CMAKE_C_COMPILER gcc-4.9)
#set(CMAKE_CXX_COMPILER g++-4.9)

#############################################
# Dependencies

#OpenCV
find_package(OpenCV REQUIRED)
#find_package(Boost REQUIRED)
find_package(Boost 1.36.0 COMPONENTS filesystem system)
if(NOT OpenCV_FOUND)
    message(SEND_ERROR "Faild to find Opencv")
    return()
else()
	file(GLOB_RECURSE video_compression_cc src/*.cc)
	file(GLOB_RECURSE video_compression_h include/*.h)
	#message(${video_compression_cc})
	#message(${video_compression_h})
	set(VIDEO_SURVEILLANCE_DEPENDENCY_DIRS ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
	set(VIDEO_SURVEILLANCE_DEPENDENCY_LIBS ${OpenCV_LIBS} ${Boost_LIBRARIES})
    message(${VIDEO_SURVEILLANCE_DEPENDENCY_DIRS})
    message(${VIDEO_SURVEILLANCE_DEPENDENCY_LIBS})
endif()
#############################################


#############################################
#General compiler flags

#add_definitions(-DDEBUG_MODE) #This shows all messages from LOG()
add_definitions(-Wall) #Enable all warnings always
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++0x -mtune=corei7 -march=corei7 -fopenmp -lpthread")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++0x -mtune=corei7 -march=corei7  ")

#############################################
#add_subdirectory(vibe)
add_subdirectory(apps)
