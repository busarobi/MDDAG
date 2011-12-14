cmake_minimum_required(VERSION 2.6)
project(MultiBoostMDP)
file(
	GLOB_RECURSE
	source_files
	srcRL/*
	)
include_directories("${PROJECT_BINARY_DIR}")
SET(BASEPATH "${CMAKE_SOURCE_DIR}/srcRL")
INCLUDE_DIRECTORIES("${BASEPATH}")
INCLUDE_DIRECTORIES("${BASEPATH}/RLTB/include/")
INCLUDE_DIRECTORIES("${BASEPATH}/multiboost/")
INCLUDE_DIRECTORIES("${BASEPATH}/torch3/core")
INCLUDE_DIRECTORIES("${BASEPATH}/torch3/include")
INCLUDE_DIRECTORIES("${BASEPATH}/newmat/")

add_executable(
	multiboostMDP
	${source_files}
	)
	
