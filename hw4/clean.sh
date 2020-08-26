ninja -t clean

if [ -f ./gmon.out ]; then
    echo "Removing gmon.out..."
    rm gmon.out
fi

if [ -f ./profile.log ]; then
    echo "Removing profile.log..."
    rm profile.log
fi

if [ -f ./nonce.out ]; then
    echo "Removing nonce.out..."
    rm nonce.out
fi
