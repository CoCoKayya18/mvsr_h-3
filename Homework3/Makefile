CC = g++
CFLAGS = -Wall -g
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

POSE_ESTIMATE_SRCS = poseEstimate/poseEstimation.cpp
PREPROCESSING_SRCS = preprocessing/preprocessing.cpp
SIFT_MATCHING_SRCS = siftMatching/siftMatching.cpp

POSE_ESTIMATE_OBJS = $(POSE_ESTIMATE_SRCS:.cpp=.o)
PREPROCESSING_OBJS = $(PREPROCESSING_SRCS:.cpp=.o)
SIFT_MATCHING_OBJS = $(SIFT_MATCHING_SRCS:.cpp=.o)

all: poseEstimate/poseEstimate preprocessing/preprocessing siftMatching/siftMatching

poseEstimate/poseEstimate: $(POSE_ESTIMATE_OBJS)
	$(CC) $(CFLAGS) -o $@ $(POSE_ESTIMATE_OBJS) $(LIBS)

preprocessing/preprocessing: $(PREPROCESSING_OBJS)
	$(CC) $(CFLAGS) -o $@ $(PREPROCESSING_OBJS) $(LIBS)

siftMatching/siftMatching: $(SIFT_MATCHING_OBJS)
	$(CC) $(CFLAGS) -o $@ $(SIFT_MATCHING_OBJS) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f poseEstimate/poseEstimate preprocessing/preprocessing siftMatching/siftMatching $(POSE_ESTIMATE_OBJS) $(PREPROCESSING_OBJS) $(SIFT_MATCHING_OBJS)
