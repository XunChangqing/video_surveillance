include_directories(${VIDEO_SURVEILLANCE_DEPENDENCY_DIRS})

##################################
# Sources

file(GLOB_RECURSE sources_cc *.cc)
file(GLOB_RECURSE sources_h *.h)
##################################

##################################
# Target

add_executable(video_speedup ${sources_cc} ${sources_h} ${video_compression_cc} ${video_compression_h})
target_link_libraries(video_speedup ${VIDEO_SURVEILLANCE_DEPENDENCY_LIBS})
##################################
