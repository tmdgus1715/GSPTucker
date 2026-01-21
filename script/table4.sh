## Tucker rank 10
./bin/P-Tucker ~/datasets/delicious-3d.tns mem_result/ 3 10 12
./bin/P-Tucker ~/datasets/nell-1.tns mem_result/ 3 10 12 
./bin/P-Tucker ~/datasets/flickr-3d.tns mem_result/ 3 10 12 
./bin/P-Tucker ~/datasets/vast-2015-mc1-3d.tns mem_result/ 3 10
./bin/P-Tucker ~/datasets/nell-2.tns mem_result/ 3 10 12 
./bin/P-Tucker ~/datasets/reddit-2015.tns mem_result/ 3 10 12
./bin/P-Tucker ~/datasets/amazon-reviews.tns mem_result/ 3 10 12
./bin/P-Tucker ~/datasets/nips.tns mem_result/ 4 10 12 
./bin/P-Tucker ~/datasets/uber.tns mem_result/ 4 10 12 
./bin/P-Tucker ~/datasets/enron.tns mem_result/ 4 10 12 
./bin/P-Tucker ~/datasets/flickr-4d.tns mem_result/ 4 10 12 
./bin/P-Tucker ~/datasets/chicago-crime-comm.tns mem_result/ 4 10 12
./bin/P-Tucker ~/datasets/chicago-crime-geo.tns mem_result/ 5 10 12
./bin/P-Tucker ~/datasets/lbnl-network.tns mem_result/ 5 10 12 

./bin/Shot --scan --size-rank 10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/nell-2.tns ./result
./bin/Shot --scan --size-rank 10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/delicious-3d.tns ./result
./bin/Shot --scan --size-rank 10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/vast-2015-mc1-3d.tns ./result
./bin/Shot --scan --size-rank 10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/flickr-3d.tns ./result
./bin/Shot --scan --size-rank 10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/nell-1.tns ./result
./bin/Shot --scan --size-rank 10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/reddit-2015.tns ./result
./bin/Shot --scan --size-rank 10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/amazon-reviews.tns ./result
./bin/Shot --scan --size-rank 10:10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/uber.tns ./result
./bin/Shot --scan --size-rank 10:10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/nips.tns ./result
./bin/Shot --scan --size-rank 10:10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/enron.tns ./result
./bin/Shot --scan --size-rank 10:10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/flickr-4d.tns ./result
./bin/Shot --scan --size-rank 10:10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/chicago-crime-comm.tns ./result
./bin/Shot --scan --size-rank 10:10:10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/lbnl-network.tns ./result
./bin/Shot --scan --size-rank 10:10:10:10:10 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/chicago-crime-geo.tns ./result

./bin/SGD__Tucker ~/datasets/delicious-3d.tns ~/datasets/delicious-3d.tns 10 3 10 10 10 
./bin/SGD__Tucker ~/datasets/nell-1.tns ~/datasets/nell-1.tns 10 3 10 10 10
./bin/SGD__Tucker ~/datasets/nell-2.tns ~/datasets/nell-2.tns 10 3 10 10 10
./bin/SGD__Tucker ~/datasets/flickr-3d.tns ~/datasets/flickr-3d.tns 10 3 10 10 10
./bin/SGD__Tucker ~/datasets/vast-2015-mc1-3d.tns ~/datasets/vast-2015-mc1-3d.tns 10 3 10 10 10
./bin/SGD__Tucker ~/datasets/uber.tns ~/datasets/uber.tns 10 4 10 10 10 10
./bin/SGD__Tucker ~/datasets/nips.tns ~/datasets/nips.tns 10 4 10 10 10 10
./bin/SGD__Tucker ~/datasets/enron.tns ~/datasets/enron.tns 10 4 10 10 10 10
./bin/SGD__Tucker ~/datasets/flickr-4d.tns ~/datasets/flickr-4d.tns 10 4 10 10 10 10
./bin/SGD__Tucker ~/datasets/chicago-crime-comm.tns ~/datasets/chicago-crime-comm.tns 10 4 10 10 10 10
./bin/SGD__Tucker ~/datasets/amazon-reviews.tns ~/datasets/amazon-reviews.tns 10 3 10 10 10
./bin/SGD__Tucker ~/datasets/reddit-2015.tns ~/datasets/reddit-2015.tns 10 3 10 10 10
./bin/SGD__Tucker ~/datasets/lbnl-network.tns ~/datasets/lbnl-network.tns 10 5 10 10 10 10 10
./bin/SGD__Tucker ~/datasets/chicago-crime-geo.tns ~/datasets/chicago-crime-geo.tns 10 5 10 10 10 10 10

./bin/FTcom --train-path ~/datasets/nell-1.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/nell-1_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/nell-2.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/nell-2_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/vast-2015-mc1-3d.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/vast-2015-mc1-3d_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/flickr-3d.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/flickr-3d_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/chicago-crime-comm.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/chicago-crime-comm_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/nips.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/nips_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/uber.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/uber_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/enron.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/enron_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/flickr-4d.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/flickr-4d_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/chicago-crime-geo.tns --tensor-order 5 --result-path /home/chon0705/ftcom/result/chicago-crime-geo_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/lbnl-network.tns --tensor-order 5 --result-path /home/chon0705/ftcom/result/lbnl-network_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/amazon-reviews.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/amazon-reviews_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/reddit-2015.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/reddit-2015_rank10 --rank-size 10 --iteration-size 1 --thread-size 12 --lambda 0.001

./bin/VeST ~/datasets/delicious-3d.tns mem_result/ 3 10 10 10 0.8 12
./bin/VeST ~/datasets/nell-1.tns mem_result/ 3 10 10 10 0.8 12
./bin/VeST ~/datasets/nell-2.tns mem_result/ 3 10 10 10 0.8 12 
./bin/VeST ~/datasets/flickr-3d.tns mem_result/ 3 10 10 10 0.8 12
./bin/VeST ~/datasets/vast-2015-mc1-3d.tns mem_result/ 3 10 10 10 0.8 12
./bin/VeST ~/datasets/uber.tns mem_result/ 4 10 10 10 10 0.8 12 
./bin/VeST ~/datasets/nips.tns mem_result/ 4 10 10 10 10 0.8 12 
./bin/VeST ~/datasets/enron.tns mem_result/ 4 10 10 10 10 0.8 12 
./bin/VeST ~/datasets/flickr-4d.tns mem_result/ 4 10 10 10 10 0.8 12 
./bin/VeST ~/datasets/chicago-crime-comm.tns mem_result/ 4 10 10 10 10 0.8 12
./bin/VeST ~/datasets/amazon-reviews.tns mem_result/ 3 10 10 10 0.8 12
./bin/VeST ~/datasets/reddit-2015.tns mem_result/ 3 10 10 10 0.8 12
./bin/VeST ~/datasets/lbnl-network.tns mem_result/ 5 10 10 10 10 10 0.8 12 
./bin/VeST ~/datasets/chicago-crime-geo.tns mem_result/ 5 10 10 10 10 10 0.8 12


./bin/PARTI --dev 12 --dims "12092,9184,28818" -l 5 ~/datasets/nell-2.tns 10 10 10 0 1 2
./bin/PARTI --dev 12 --dims "1605,4198,1631,4209,868131" -l 5 ~/datasets/lbnl-network.tns 10 10 10 10 10 0 1 2 3 4
./bin/PARTI --dev 12 --dims "6185,24,380,395,32" -l 5 ~/datasets/chicago-crime-geo.tns 10 10 10 10 10 0 1 2 3 4
./bin/PARTI --dev 12 --dims "165427,11374,2" -l 5 ~/datasets/vast-2015-mc1-3d.tns 10 10 10 0 1 2
./bin/PARTI --dev 12 --dims "183,24,1140,1717" -l 5 ~/datasets/uber.tns 10 10 10 10 0 1 2 3
./bin/PARTI --dev 12 --dims "2482,2862,14036,17" -l 5 ~/datasets/nips.tns 10 10 10 10 0 1 2 3
./bin/PARTI --dev 12 --dims "6066,5699,244268,1176" -l 5 ~/datasets/enron.tns 10 10 10 10 0 1 2 3
./bin/PARTI --dev 12 --dims "6186,24,77,32" -l 5 ~/datasets/chicago-crime-comm.tns 10 10 10 10 0 1 2 3

./bin/GTA ~/datasets/delicious-3d.tns mem_result/ 10 2
./bin/GTA ~/datasets/nell-1.tns mem_result/ 10 2
./bin/GTA ~/datasets/flickr-3d.tns mem_result/ 10 2
./bin/GTA ~/datasets/vast-2015-mc1-3d.tns mem_result/ 10 2
./bin/GTA ~/datasets/uber.tns mem_result/ 10 2 
./bin/GTA ~/datasets/nips.tns mem_result/ 10 2 
./bin/GTA ~/datasets/nell-2.tns mem_result/ 10 2
./bin/GTA ~/datasets/enron.tns mem_result/ 10 2 
./bin/GTA ~/datasets/flickr-4d.tns mem_result/ 10 2 
./bin/GTA ~/datasets/chicago-crime-comm.tns mem_result/ 10 2
./bin/GTA ~/datasets/lbnl-network.tns mem_result/ 10 2 
./bin/GTA ~/datasets/chicago-crime-geo.tns mem_result/ 10 2
./bin/GTA ~/datasets/amazon-reviews.tns mem_result/ 10 2
./bin/GTA ~/datasets/reddit-2015.tns mem_result/ 10 2

./bin/GPUTucker -i ~/datasets/delicious-3d.tns -o 3 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/nell-1.tns -o 3 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/flickr-3d.tns -o 3 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/nell-2.tns -o 3 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/vast-2015-mc1-3d.tns -o 3 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/chicago-crime-comm.tns -o 4 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/nips.tns -o 4 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/uber.tns -o 4 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/enron.tns -o 4 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/flickr-4d.tns -o 4 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/chicago-crime-geo.tns -o 5 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/lbnl-network.tns -o 5 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/amazon-reviews.tns -o 3 -r 10 -g 2
./bin/GPUTucker -i ~/datasets/reddit-2015.tns -o 3 -r 10 -g 2

./bin/GSPTucker -i ~/datasets/nell-2.tns -o 3 -r 10 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/delicious-3d.tns -o 3 -r 10 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/vast-2015-mc1-3d.tns -o 3 -r 10 -g 2 -c 4  -H 48
./bin/GSPTucker -i ~/datasets/nips.tns -o 4 -r 10 -g 2 -c 4  -H 48
./bin/GSPTucker -i ~/datasets/enron.tns -o 4 -r 10 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/lbnl-network.tns -o 5 -r 10 -g 2 -c 4  -H 48
./bin/GSPTucker -i ~/datasets/nell-1.tns -o 3 -r 10 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/uber.tns -o 4 -r 10 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/chicago-crime-geo.tns -o 5 -r 10 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/chicago-crime-comm.tns -o 4 -r 10 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/flickr-4d.tns -o 4 -r 10 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/amazon-reviews.tns -o 3 -r 10 -g 2 -c 4 -H 48 -a
./bin/GSPTucker -i ~/datasets/reddit-2015.tns -o 3 -r 10 -g 2 -c 4 -H 48 -a


## Tucker rank 20 
./bin/P-Tucker ~/datasets/delicious-3d.tns mem_result/ 3 20 12
./bin/P-Tucker ~/datasets/nell-1.tns mem_result/ 3 20 12 
./bin/P-Tucker ~/datasets/flickr-3d.tns mem_result/ 3 20 12 
./bin/P-Tucker ~/datasets/vast-2015-mc1-3d.tns mem_result/ 3 20
./bin/P-Tucker ~/datasets/nell-2.tns mem_result/ 3 20 12 
./bin/P-Tucker ~/datasets/reddit-2015.tns mem_result/ 3 20 12
./bin/P-Tucker ~/datasets/amazon-reviews.tns mem_result/ 3 20 12
./bin/P-Tucker ~/datasets/nips.tns mem_result/ 4 20 12 
./bin/P-Tucker ~/datasets/uber.tns mem_result/ 4 20 12 
./bin/P-Tucker ~/datasets/enron.tns mem_result/ 4 20 12 
./bin/P-Tucker ~/datasets/flickr-4d.tns mem_result/ 4 20 12 
./bin/P-Tucker ~/datasets/chicago-crime-comm.tns mem_result/ 4 20 12
./bin/P-Tucker ~/datasets/chicago-crime-geo.tns mem_result/ 5 20 12
./bin/P-Tucker ~/datasets/lbnl-network.tns mem_result/ 5 20 12 

./bin/Shot --scan --size-rank 20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/nell-2.tns ./result
./bin/Shot --scan --size-rank 20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/delicious-3d.tns ./result
./bin/Shot --scan --size-rank 20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/vast-2015-mc1-3d.tns ./result
./bin/Shot --scan --size-rank 20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/flickr-3d.tns ./result
./bin/Shot --scan --size-rank 20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/nell-1.tns ./result
./bin/Shot --scan --size-rank 20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/reddit-2015.tns ./result
./bin/Shot --scan --size-rank 20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/amazon-reviews.tns ./result
./bin/Shot --scan --size-rank 20:20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/uber.tns ./result
./bin/Shot --scan --size-rank 20:20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/nips.tns ./result
./bin/Shot --scan --size-rank 20:20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/enron.tns ./result
./bin/Shot --scan --size-rank 20:20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/flickr-4d.tns ./result
./bin/Shot --scan --size-rank 20:20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/chicago-crime-comm.tns ./result
./bin/Shot --scan --size-rank 20:20:20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/lbnl-network.tns ./result
./bin/Shot --scan --size-rank 20:20:20:20:20 --use-coreloss --use-iteration --iteration 1 --size-thread 12 ~/datasets/chicago-crime-geo.tns ./result

./bin/SGD__Tucker ~/datasets/delicious-3d.tns ~/datasets/delicious-3d.tns 20 3 20 20 20 
./bin/SGD__Tucker ~/datasets/nell-1.tns ~/datasets/nell-1.tns 20 3 20 20 20
./bin/SGD__Tucker ~/datasets/nell-2.tns ~/datasets/nell-2.tns 20 3 20 20 20
./bin/SGD__Tucker ~/datasets/flickr-3d.tns ~/datasets/flickr-3d.tns 20 3 20 20 20
./bin/SGD__Tucker ~/datasets/vast-2015-mc1-3d.tns ~/datasets/vast-2015-mc1-3d.tns 20 3 20 20 20
./bin/SGD__Tucker ~/datasets/uber.tns ~/datasets/uber.tns 20 4 20 20 20 20
./bin/SGD__Tucker ~/datasets/nips.tns ~/datasets/nips.tns 20 4 20 20 20 20
./bin/SGD__Tucker ~/datasets/enron.tns ~/datasets/enron.tns 20 4 20 20 20 20
./bin/SGD__Tucker ~/datasets/flickr-4d.tns ~/datasets/flickr-4d.tns 20 4 20 20 20 20
./bin/SGD__Tucker ~/datasets/chicago-crime-comm.tns ~/datasets/chicago-crime-comm.tns 20 4 20 20 20 20
./bin/SGD__Tucker ~/datasets/amazon-reviews.tns ~/datasets/amazon-reviews.tns 20 3 20 20 20
./bin/SGD__Tucker ~/datasets/reddit-2015.tns ~/datasets/reddit-2015.tns 20 3 20 20 20
./bin/SGD__Tucker ~/datasets/lbnl-network.tns ~/datasets/lbnl-network.tns 20 5 20 20 20 20 20
./bin/SGD__Tucker ~/datasets/chicago-crime-geo.tns ~/datasets/chicago-crime-geo.tns 20 5 20 20 20 20 20

./bin/FTcom --train-path ~/datasets/delicious-3d.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/delicious-3d_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/nell-1.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/nell-1_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/nell-2.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/nell-2_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/vast-2015-mc1-3d.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/vast-2015-mc1-3d_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/flickr-3d.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/flickr-3d_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/chicago-crime-comm.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/chicago-crime-comm_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/nips.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/nips_rank20 --rank-size 20 --iteration-size 1 --thread-size 12 --lambda 0.001
./bin/FTcom --train-path ~/datasets/uber.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/uber_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/enron.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/enron_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/flickr-4d.tns --tensor-order 4 --result-path /home/chon0705/ftcom/result/flickr-4d_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/chicago-crime-geo.tns --tensor-order 5 --result-path /home/chon0705/ftcom/result/chicago-crime-geo_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/lbnl-network.tns --tensor-order 5 --result-path /home/chon0705/ftcom/result/lbnl-network_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/amazon-reviews.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/amazon-reviews_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001
./bin/FTcom --train-path ~/datasets/reddit-2015.tns --tensor-order 3 --result-path /home/chon0705/ftcom/result/reddit-2015_rank20 --rank-size 20 --iteration-size 1 --thread-size 40 --lambda 0.001

./bin/VeST ~/datasets/delicious-3d.tns mem_result/ 3 20 20 20 0.8 12
./bin/VeST ~/datasets/nell-1.tns mem_result/ 3 20 20 20 0.8 12
./bin/VeST ~/datasets/nell-2.tns mem_result/ 3 20 20 20 0.8 12 
./bin/VeST ~/datasets/flickr-3d.tns mem_result/ 3 20 20 20 0.8 12
./bin/VeST ~/datasets/vast-2015-mc1-3d.tns mem_result/ 3 20 20 20 0.8 12
./bin/VeST ~/datasets/uber.tns mem_result/ 4 20 20 20 20 0.8 12 
./bin/VeST ~/datasets/nips.tns mem_result/ 4 20 20 20 20 0.8 12 
./bin/VeST ~/datasets/enron.tns mem_result/ 4 20 20 20 20 0.8 12 
./bin/VeST ~/datasets/flickr-4d.tns mem_result/ 4 20 20 20 20 0.8 12 
./bin/VeST ~/datasets/chicago-crime-comm.tns mem_result/ 4 20 20 20 20 0.8 12
./bin/VeST ~/datasets/amazon-reviews.tns mem_result/ 3 20 20 20 0.8 12
./bin/VeST ~/datasets/reddit-2015.tns mem_result/ 3 20 20 20 0.8 12
./bin/VeST ~/datasets/lbnl-network.tns mem_result/ 5 20 20 20 20 20 0.8 12 
./bin/VeST ~/datasets/chicago-crime-geo.tns mem_result/ 5 20 20 20 20 20 0.8 12


./bin/PARTI --dev 12 --dims "12092,9184,28818" -l 5 ~/datasets/nell-2.tns 20 20 20 0 1 2
./bin/PARTI --dev 12 --dims "1605,4198,1631,4209,868131" -l 5 ~/datasets/lbnl-network.tns 20 20 20 20 20 0 1 2 3 4
./bin/PARTI --dev 12 --dims "6185,24,380,395,32" -l 5 ~/datasets/chicago-crime-geo.tns 20 20 20 20 20 0 1 2 3 4
./bin/PARTI --dev 12 --dims "165427,11374,2" -l 5 ~/datasets/vast-2015-mc1-3d.tns 20 20 20 0 1 2
./bin/PARTI --dev 12 --dims "183,24,1140,1717" -l 5 ~/datasets/uber.tns 20 20 20 20 0 1 2 3
./bin/PARTI --dev 12 --dims "2482,2862,14036,17" -l 5 ~/datasets/nips.tns 20 20 20 20 0 1 2 3
./bin/PARTI --dev 12 --dims "6066,5699,244268,1176" -l 5 ~/datasets/enron.tns 20 20 20 20 0 1 2 3
./bin/PARTI --dev 12 --dims "6186,24,77,32" -l 5 ~/datasets/chicago-crime-comm.tns 20 20 20 20 0 1 2 3

./bin/GTA ~/datasets/delicious-3d.tns mem_result/ 20 2
./bin/GTA ~/datasets/nell-1.tns mem_result/ 20 2
./bin/GTA ~/datasets/flickr-3d.tns mem_result/ 20 2
./bin/GTA ~/datasets/vast-2015-mc1-3d.tns mem_result/ 20 2
./bin/GTA ~/datasets/uber.tns mem_result/ 20 2 
./bin/GTA ~/datasets/nips.tns mem_result/ 20 2 
./bin/GTA ~/datasets/nell-2.tns mem_result/ 20 2
./bin/GTA ~/datasets/enron.tns mem_result/ 20 2 
./bin/GTA ~/datasets/flickr-4d.tns mem_result/ 20 2 
./bin/GTA ~/datasets/chicago-crime-comm.tns mem_result/ 20 2
./bin/GTA ~/datasets/lbnl-network.tns mem_result/ 20 2 
./bin/GTA ~/datasets/chicago-crime-geo.tns mem_result/ 20 2
./bin/GTA ~/datasets/amazon-reviews.tns mem_result/ 20 2
./bin/GTA ~/datasets/reddit-2015.tns mem_result/ 20 2

./bin/GPUTucker -i ~/datasets/delicious-3d.tns -o 3 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/nell-1.tns -o 3 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/flickr-3d.tns -o 3 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/nell-2.tns -o 3 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/vast-2015-mc1-3d.tns -o 3 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/chicago-crime-comm.tns -o 4 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/nips.tns -o 4 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/uber.tns -o 4 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/enron.tns -o 4 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/flickr-4d.tns -o 4 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/chicago-crime-geo.tns -o 5 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/lbnl-network.tns -o 5 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/amazon-reviews.tns -o 3 -r 20 -g 2
./bin/GPUTucker -i ~/datasets/reddit-2015.tns -o 3 -r 20 -g 2

./bin/GSPTucker -i ~/datasets/nell-2.tns -o 3 -r 20 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/delicious-3d.tns -o 3 -r 20 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/vast-2015-mc1-3d.tns -o 3 -r 20 -g 2 -c 4  -H 48
./bin/GSPTucker -i ~/datasets/nips.tns -o 4 -r 20 -g 2 -c 4  -H 48
./bin/GSPTucker -i ~/datasets/enron.tns -o 4 -r 20 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/lbnl-network.tns -o 5 -r 20 -g 2 -c 4  -H 48
./bin/GSPTucker -i ~/datasets/nell-1.tns -o 3 -r 20 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/uber.tns -o 4 -r 20 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/chicago-crime-geo.tns -o 5 -r 20 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/chicago-crime-comm.tns -o 4 -r 20 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/flickr-4d.tns -o 4 -r 20 -g 2 -c 4  -H 48 -a
./bin/GSPTucker -i ~/datasets/amazon-reviews.tns -o 3 -r 20 -g 2 -c 4 -H 48 -a
./bin/GSPTucker -i ~/datasets/reddit-2015.tns -o 3 -r 20 -g 2 -c 4 -H 48 -a
