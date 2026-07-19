LOCALHOST=`hostname -s`
if [ $LOCALHOST = aluf ]; then kill -9 113105; else ssh aluf kill -9 113105; fi
if [ $LOCALHOST = aluf ]; then kill -9 113103; else ssh aluf kill -9 113103; fi
if [ $LOCALHOST = aluf ]; then kill -9 113111; else ssh aluf kill -9 113111; fi
if [ $LOCALHOST = aluf ]; then kill -9 113117; else ssh aluf kill -9 113117; fi

rm -f cleanup-ansys-aluf-113117.sh
