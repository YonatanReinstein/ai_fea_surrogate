LOCALHOST=`hostname -s`
if [ $LOCALHOST = aluf ]; then kill -9 121998; else ssh aluf kill -9 121998; fi
if [ $LOCALHOST = aluf ]; then kill -9 122005; else ssh aluf kill -9 122005; fi
if [ $LOCALHOST = aluf ]; then kill -9 122034; else ssh aluf kill -9 122034; fi
if [ $LOCALHOST = aluf ]; then kill -9 122036; else ssh aluf kill -9 122036; fi

rm -f cleanup-ansys-aluf-122036.sh
