LOCALHOST=`hostname -s`
if [ $LOCALHOST = aluf ]; then kill -9 113176; else ssh aluf kill -9 113176; fi
if [ $LOCALHOST = aluf ]; then kill -9 113187; else ssh aluf kill -9 113187; fi
if [ $LOCALHOST = aluf ]; then kill -9 113190; else ssh aluf kill -9 113190; fi
if [ $LOCALHOST = aluf ]; then kill -9 113189; else ssh aluf kill -9 113189; fi

rm -f cleanup-ansys-aluf-113189.sh
