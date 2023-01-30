# FedDefender: Client-Side Attack-Tolerant Federated Learning 

Pytorch Implementation of FedDefender: Client-Side Attack-Tolerant Federated Learning
* Unique client-side defense strategy, FedDefender, that can train robust local models against malicious attacks from adversaries
* Designing an attack-tolerant local meta update that helps discover noise-tolerant parameters for local models by utilizing a synthetically corrupted training set.
* Introducing an attack-tolerant global knowledge distillation technique that efficiently aligns the local model’s knowledge to the global data distribution while reducing the negative effects of false information in the possibly-corrupted global model


## Model architecture ##
<center><img src="./fig/model.png"> </center>

## Usage ##
```
usage: main.py [-h] [--model MODEL] [--dataset DATASET] [--net_config NET_CONFIG] [--partition PARTITION] [--batch-size BATCH_SIZE]
               [--lr LR] [--epochs EPOCHS] [--n_parties N_PARTIES] [--n_parties_per_round N_PARTIES_PER_ROUND] [--alg ALG]
               [--comm_round COMM_ROUND] [--init_seed INIT_SEED] [--dropout_p DROPOUT_P] [--datadir DATADIR] [--reg REG]
               [--logdir LOGDIR] [--modeldir MODELDIR] [--beta BETA] [--device DEVICE] [--log_file_name LOG_FILE_NAME]
               [--optimizer OPTIMIZER] [--temperature TEMPERATURE] [--attacker_type ATTACKER_TYPE] [--attacker_ratio ATTACKER_RATIO]
               [--noise_ratio NOISE_RATIO] [--global_defense GLOBAL_DEFENSE]

Parser

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         neural network used in training
  --dataset DATASET     dataset used for training
  --net_config NET_CONFIG
  --partition PARTITION
                        the data partitioning strategy
  --batch-size BATCH_SIZE
                        input batch size for training 
  --lr LR               learning rate 
  --epochs EPOCHS       number of local epochs
  --n_parties N_PARTIES
                        number of workers in a distributed cluster
  --n_parties_per_round N_PARTIES_PER_ROUND
                        number of workers in a distributed cluster per round
  --alg ALG             training strategy: defender/vanilla
  --comm_round COMM_ROUND
                        number of maximum communication roun
  --init_seed INIT_SEED
                        Random seed
  --dropout_p DROPOUT_P
                        Dropout probability. 
  --datadir DATADIR     Data directory
  --reg REG             L2 regularization strength
  --logdir LOGDIR       Log directory path
  --modeldir MODELDIR   Model directory path
  --beta BETA           The parameter for the dirichlet distribution for data partitioning
  --device DEVICE       The device to run the program
  --log_file_name LOG_FILE_NAME
                        The log file name
  --optimizer OPTIMIZER
                        the optimizer
  --temperature TEMPERATURE
                        the temperature parameter for knowledge distillation
  --attacker_type ATTACKER_TYPE
                        attacker type (either untargeted or untargeted_diverse)
  --attacker_ratio ATTACKER_RATIO
                        ratio for number of attackers
  --noise_ratio NOISE_RATIO
                        noise ratio for label flipping (0 to 1)
  --global_defense GLOBAL_DEFENSE
                        communication strategy: average/median/krum/norm/residual
```

