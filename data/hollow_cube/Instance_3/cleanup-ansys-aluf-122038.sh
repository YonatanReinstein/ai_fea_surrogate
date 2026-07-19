LOCALHOST=`hostname -s`
if [ $LOCALHOST = aluf ]; then kill -9 122018; else ssh aluf kill -9 122018; fi
if [ $LOCALHOST = aluf ]; then kill -9 122001; else ssh aluf kill -9 122001; fi
if [ $LOCALHOST = aluf ]; then kill -9 122040; else ssh aluf kill -9 122040; fi
if [ $LOCALHOST = aluf ]; then kill -9 122038; else ssh aluf kill -9 122038; fi

rm -f cleanup-ansys-aluf-122038.sh
