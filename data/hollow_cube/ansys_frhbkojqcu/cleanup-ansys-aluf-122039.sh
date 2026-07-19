LOCALHOST=`hostname -s`
if [ $LOCALHOST = aluf ]; then kill -9 122011; else ssh aluf kill -9 122011; fi
if [ $LOCALHOST = aluf ]; then kill -9 122010; else ssh aluf kill -9 122010; fi
if [ $LOCALHOST = aluf ]; then kill -9 122029; else ssh aluf kill -9 122029; fi
if [ $LOCALHOST = aluf ]; then kill -9 122039; else ssh aluf kill -9 122039; fi

rm -f cleanup-ansys-aluf-122039.sh
