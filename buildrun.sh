#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

ant

sudo -u hduser /usr/local/hadoop/bin/hadoop dfs -copyFromLocal $DIR/input /user/hduser/input

sudo -u hduser /usr/local/hadoop/bin/hadoop dfs -rmr /user/hduser/method-otsu-out

sudo -u hduser /usr/local/hadoop/bin/hadoop jar /develop/java/NewSandCastle/MethodOzuTest.jar /user/hduser/input /user/hduser/method-otsu-out
