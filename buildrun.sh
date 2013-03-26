#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INPUT_DFS_DIR=/user/hduser/method-otsu-in
TMP_DFS_DIR=/user/hduser/tmp
OUTPUT_DFS_DIR=/user/hduser/method-otsu-out
echo $DIR
echo $HADOOP_HOME

ant

sudo -u hduser $HADOOP_HOME/bin/hadoop dfs -rmr $INPUT_DFS_DIR
sudo -u hduser $HADOOP_HOME/bin/hadoop dfs -rmr $TMP_DFS_DIR
sudo -u hduser $HADOOP_HOME/bin/hadoop dfs -rmr $OUTPUT_DFS_DIR

sudo -u hduser $HADOOP_HOME/bin/hadoop dfs -copyFromLocal $DIR/input $INPUT_DFS_DIR

sudo -u hduser $HADOOP_HOME/bin/hadoop jar $DIR/MethodOzu.jar $INPUT_DFS_DIR $OUTPUT_DFS_DIR
