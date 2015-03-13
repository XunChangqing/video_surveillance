##################################
# vibe simple app
##################################

include_directories(${VIDEO_SURVEILLANCE_DEPENDENCY_DIRS})

##################################
# Sources

file(GLOB_RECURSE sources_cc *.cc)
file(GLOB_RECURSE sources_h *.h)
##################################

##################################
# Target

add_executable(video_wrap ${sources_cc} ${sources_h} ${video_compression_cc} ${video_compression_h})
target_link_libraries(video_wrap ${VIDEO_SURVEILLANCE_DEPENDENCY_LIBS})
##################################
