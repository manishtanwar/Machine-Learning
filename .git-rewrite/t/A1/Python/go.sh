if [ $1 == "1" ]
then
	./run.sh 1 ../ass1_data1/linearX.csv ../ass1_data1/linearY.csv $2 0.2
elif [ $1 == "2" ]
then
	./run.sh 2 ../ass1_data1/weightedX.csv ../ass1_data1/weightedY.csv 0.2
elif [ $1 == "3" ]
then
	./run.sh 3 ../ass1_data1/logisticX.csv ../ass1_data1/logisticY.csv
else
	./run.sh 4 ../ass1_data1/q4x.dat ../ass1_data1/q4y.dat 1
fi
