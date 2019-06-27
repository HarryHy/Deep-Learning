argparser = argparse.ArgumentParser()
argparser.add_argument('-m', '--model_dir', type=str, default="./model/", help="path of the directory to store model files")
argparser.add_argument('-f', '--form', type=str, default="./", help="path of the directory of result form files that store the status of every epoch")

argparser.add_argument('-d', '--dataset', type=str, choices=["SVHN", "CIFAR-10", "CIFAR-100", "None"], default="None", help="dataset to use")
argparser.add_argument('-ce', '--cuda_enabled', action='store_true', default=False, help="use the cuda or not")

argparser.add_argument('-k', '--num_cluster', type=int, default=1)
argparser.add_argument('-lrd', '--learning_rate_decay', type=int, nargs=2, default=(1, 1), help="step and rate of learning rate decay method")
