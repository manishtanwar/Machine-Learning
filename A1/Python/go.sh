if [ $1 == "1" ]
then
	./run.sh 1 ../ass1_data/linearX.csv ../ass1_data/linearY.csv 1.6 0.2
elif [ $1 == "2" ]
then
	./run.sh 2 ../ass1_data/weightedX.csv ../ass1_data/weightedY.csv 0.8
elif [ $1 == "3" ]
then
	./run.sh 3 ../ass1_data/logisticX.csv ../ass1_data/logisticY.csv
else
	./run.sh 4 ../ass1_data/q4x.dat ../ass1_data/q4y.dat 1
fi
