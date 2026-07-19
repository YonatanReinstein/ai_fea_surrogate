LOCALHOST=`hostname -s`
if [ $LOCALHOST = aluf ]; then kill -9 121988; else ssh aluf kill -9 121988; fi
if [ $LOCALHOST = aluf ]; then kill -9 121949; else ssh aluf kill -9 121949; fi
if [ $LOCALHOST = aluf ]; then kill -9 122022; else ssh aluf kill -9 122022; fi
if [ $LOCALHOST = aluf ]; then kill -9 122033; else ssh aluf kill -9 122033; fi

rm -f cleanup-ansys-aluf-122033.sh
