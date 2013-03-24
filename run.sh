#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

sudo -u hduser /usr/local/hadoop/bin/hadoop dfs -rmr /user/hduser/method-otsu-in
sudo -u hduser /usr/local/hadoop/bin/hadoop dfs -rmr /user/hduser/method-otsu-out

sudo -u hduser /usr/local/hadoop/bin/hadoop dfs -copyFromLocal $DIR/input /user/hduser/in

sudo -u hduser /usr/local/hadoop/bin/hadoop jar $DIR/MethodOzuTest.jar /user/hduser/in /user/hduser/method-otsu-out
