include_directories(${VIDEO_SURVEILLANCE_DEPENDENCY_DIRS})

##################################
# Sources

file(GLOB_RECURSE sources_cc *.cc)
file(GLOB_RECURSE sources_h *.h)
##################################

##################################
# Target

add_executable(bgs_app ${sources_cc} ${sources_h} ${video_compression_cc} ${video_compression_h})
target_link_libraries(bgs_app ${VIDEO_SURVEILLANCE_DEPENDENCY_LIBS})
##################################
