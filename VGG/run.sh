echo "offset    time"
for i in {0..10}
do
        ./VGG -o $[i*10]
        sleep 5
done