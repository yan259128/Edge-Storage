


echo "starting keys generation" &&
sawadm keygen validator-1 &&
sawadm keygen validator-2 &&
sawadm keygen validator-3 &&
#sawadm keygen validator-4 &&
#sawadm keygen validator-5 &&
#sawadm keygen validator-6 &&
#sawadm keygen validator-7 &&
#sawadm keygen validator-8 &&
#sawadm keygen validator-9 &&
#sawadm keygen validator-10 &&
#sawadm keygen validator-11 &&
sawadm keygen &&
echo "done with keys generation"



#for i in {1..10}; do
#    key_file="/etc/sawtooth/keys/validator-${i}.priv"
#    batch_file="/poet_keys/poet-${i}.batch"
#
#    # 检查私钥文件是否存在
#    if [ -f "$key_file" ]; then
#        echo "Generating poet batch for validator-${i}"
#        poet registration create --key "$key_file" -o "$batch_file"
#    else
#        echo "Key file for validator-${i} not found: $key_file"
#    fi
#done
mv /etc/sawtooth/keys/validator-* /poet_keys &&
cp /etc/sawtooth/keys/validator.* /poet_keys

if [ -f "/etc/sawtooth/keys/validator.priv" ]; then
    poet registration create --key "/etc/sawtooth/keys/validator.priv" -o "/poet_keys/poet.batch"
  else
    echo "Key file for validator not found: /etc/sawtooth/keys/validator.priv"
fi



poet enclave measurement --enclave-module simulator
poet enclave basename --enclave-module simulator

poet enclave measurement --enclave-module simulator > valid_enclave_measurements.txt
poet enclave basename --enclave-module simulator > valid_enclave_basenames.txt

mv valid_enclave_basenames.txt /poet_keys
mv valid_enclave_measurements.txt /poet_keys
cp /etc/sawtooth/simulator_rk_pub.pem /poet_keys

ls /poet_keys


echo "Batch generation complete."

while true; do
  if [ -e /shared_keys/validator.priv  ]; then
      echo "Starting PoET engine in the background..."
      poet-engine -vv --connect tcp://172.18.10.2:5050
      break ;
  fi
  sleep 1
done

