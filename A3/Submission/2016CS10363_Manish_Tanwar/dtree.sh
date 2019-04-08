if [ $1 == "1" ]
then
	g++ 1/a/prep.cpp -o prep
	./prep $2 $2 train
	./prep $2 $3 test
	./prep $2 $4 val
	g++ 1/a/1.cpp -o 1a
	./1a train test val
elif [ $1 == "2" ]
then
	g++ 1/b/b.cpp -o b
	./b $2 $3 $4
elif [ $1 == "3" ]
then
	g++ 1/c/c.cpp -o c
	./c $2 $3 $4
elif [ $1 == "4" ]
then
	python3 1/def/d.py $2 $3 $4
elif [ $1 == "5" ]
then
	python3 1/def/e.py $2 $3 $4
else
	python3 1/def/f.py $2 $3 $4
fi