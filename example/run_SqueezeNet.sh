for i in {0..10}
do
        ../bin/SqueezeNet -o $[i*10] -n 16
        sleep 10
done