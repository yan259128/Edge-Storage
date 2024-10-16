echo "starting keys generation" &&
#sawadm keygen validator-1 &&
#sawadm keygen validator-2 &&
#sawadm keygen validator-3 &&
#sawadm keygen validator-4 &&
#sawadm keygen validator-5 &&
#sawadm keygen validator-6 &&
#sawadm keygen validator-7 &&
#sawadm keygen validator-8 &&
#sawadm keygen validator-9 &&
#sawadm keygen validator-10 &&
#sawadm keygen validator-11 &&
#sawadm keygen validator-12 &&
#sawadm keygen validator-13 &&
#sawadm keygen validator-14 &&
#sawadm keygen validator-15 &&
#sawadm keygen validator-16 &&
#sawadm keygen validator-17 &&
#sawadm keygen validator-18 &&
#sawadm keygen validator-19 &&
#sawadm keygen validator-20 &&
#sawadm keygen validator-21 &&
#sawadm keygen validator-22 &&
#sawadm keygen validator-23 &&
#sawadm keygen validator-24 &&
#sawadm keygen &&

sawtooth keygen node0 &&
sawtooth keygen node1 &&
sawtooth keygen node2 &&
sawtooth keygen node3 &&
sawtooth keygen node4 &&
sawtooth keygen node5 &&
sawtooth keygen node6 &&
sawtooth keygen node7 &&
sawtooth keygen node8 &&
sawtooth keygen node9 &&
sawtooth keygen node10 &&
sawtooth keygen node11 &&
#sawtooth keygen node12 &&
#sawtooth keygen node13 &&
#sawtooth keygen node14 &&
#sawtooth keygen node15 &&
#sawtooth keygen node16 &&
#sawtooth keygen node17 &&
#sawtooth keygen node18 &&
#sawtooth keygen node19 &&
#sawtooth keygen node20 &&
#sawtooth keygen node21 &&
#sawtooth keygen node22 &&
#sawtooth keygen node23 &&
#sawtooth keygen node24 &&
echo "done with keys generation"

# poet 情况
while true; do
  if [ -e /poet_keys/poet.batch  ] && [ -e /poet_keys/simulator_rk_pub.pem ]; then
      sleep 10
      ls /poet_keys/

      cp /poet_keys/validator.* /etc/sawtooth/keys
      cp /poet_keys/validator-* /etc/sawtooth/keys

      cp /poet_keys/simulator_rk_pub.pem /etc/sawtooth/
      cp /poet_keys/valid_enclave_measurements.txt /etc/sawtooth/
      cp /poet_keys/valid_enclave_basenames.txt /etc/sawtooth/
      break ;
  fi
  sleep 1
done

ls /etc/sawtooth

# pbft 情况
#MEMBERS=$(ls /etc/sawtooth/keys/validator*.pub | xargs -n 1 cat | sed ':a;N;$!ba;s/\n/","/g' | awk '{print "[\"" $0 "\"]"}')
#MEMBERS="'$MEMBERS'"

set -x


sawset genesis \
  -k /etc/sawtooth/keys/validator.priv \
  -o config-genesis.batch &&
sawset proposal create \
  -k /etc/sawtooth/keys/validator.priv \
  sawtooth.consensus.algorithm.name=PoET \
  sawtooth.consensus.algorithm.version=0.1 \
  sawtooth.poet.report_public_key_pem="$(cat /etc/sawtooth/simulator_rk_pub.pem)" \
  sawtooth.poet.valid_enclave_measurements="$(cat /etc/sawtooth/valid_enclave_measurements.txt)" \
  sawtooth.poet.valid_enclave_basenames="$(cat /etc/sawtooth/valid_enclave_basenames.txt)" \
  -o config.batch &&
sawset proposal create --key /etc/sawtooth/keys/validator.priv \
  -o poet-settings.batch \
  sawtooth.poet.target_wait_time=1 \
  sawtooth.poet.initial_wait_time=20 \
  sawtooth.publisher.max_batches_per_block=20
#  sawtooth.consensus.pbft.members='["02a7c341a3fb5f447e07229576247736707c544ef2b2a09631f971023143d0511c","0304686f6ce93762995086a5f3135665c22e9da5f349d86987f703b735bbc2e1e4","02f70aeb669299c77b641e1b251bd33a313c8d4310a45b6d3aba484086fdb2c179","0337cce85bc6e6466287c9c9ee1261f59caff095fa14ad435dd6f42910cd902045"]' \

#echo $MEMBERS

#  sawtooth.consensus.pbft.members=$MEMBERS \

#sawadm genesis \
#  config-genesis.batch \
#  config.batch

# 判断是否存在 poet-*.batch 文件 /poet_keys/poet-*.batch
if ls /poet_keys/poet*.batch 1> /dev/null 2>&1; then
    echo "Found poet batch files, adding them to genesis"
    sawadm genesis config-genesis.batch config.batch /poet_keys/poet*.batch poet-settings.batch
else
    echo "No poet batch files found, proceeding without them"
    sawadm genesis config-genesis.batch config.batch
fi

mv /etc/sawtooth/keys/validator-* /shared_keys &&
cp /etc/sawtooth/keys/validator.* /shared_keys &&
cat /etc/sawtooth/keys/validator.pub &&
mv /root/.sawtooth/keys/node* /shared_keys


#sawtooth-validator -vvv \
#  --endpoint tcp://validator-0:8800 \
#  --bind component:tcp://eth0:4004 \
#  --bind network:tcp://eth0:8800 \
#  --bind consensus:tcp://eth0:5050 \
#  --maximum-peer-connectivity 30
#set +x


#  --peers tcp://172.18.10.3:8800,tcp://172.18.10.4:8800,tcp://172.18.10.5:8800 \

sawtooth-validator -vvv \
  --endpoint tcp://validator-0:8800 \
  --bind component:tcp://eth0:4004 \
  --bind network:tcp://eth0:8800 \
  --bind consensus:tcp://eth0:5050 \
  --peering static \
  --scheduler parallel \
  --maximum-peer-connectivity 30
set +x
