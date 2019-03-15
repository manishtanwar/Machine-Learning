if [ $1 == "1" ]
then
	python3 1/1.py $2 $3 $4
else
	if [ $4 == "0" ]
	then
		python3 2/2.py $2 $3 $4 $5
	else
		python3 2/2_multi.py $2 $3 $4 $5
	fi 
fi