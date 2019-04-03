if [ $1 == "1" ]
then
	python3 assignment1_1.py $2 $3 $4 $5
elif [ $1 == "2" ]
then
	python3 assignment1_2.py $2 $3 $4
elif [ $1 == "3" ]
then
	python3 assignment1_3.py $2 $3
else
	python3 assignment1_4.py $2 $3 $4
fi
