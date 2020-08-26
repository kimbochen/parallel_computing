echo "Building ..."
ninja cuda_miner

echo "Executing ..."
time srun -ppp --gres=gpu:1 -N1 -n1 cuda_miner testcase/00/case00.in nonce.out

echo "Checking nonce ..."
echo `diff nonce.out testcase/00/case00.out`
