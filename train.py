import argparse
from Trainer import Trainer
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported bool string format.')

def main():
    parser = argparse.ArgumentParser(description="AerialSeg by PyTorch: train.py")
    parser.add_argument('--mode', type=str, default='train', choices=['train','val'], help='which mode to run')
    parser.add_argument('--data_path', type=str, default='~/datasets', help='path of datasets')
    parser.add_argument('--train_batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--train_crop_size', type=int, default=512, help='crop size of training')
    parser.add_argument('--eval_crop_size', type=int, default=1024, help='crop size of testing')
    parser.add_argument('--stride', type=int, default=512, help='stride of testing')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from')
    parser.add_argument('--finetune', type=str, default=None, help='checkpoint to finetune')
    parser.add_argument('--cuda', type=str2bool, default=False, help='whether to use GPU')
    parser.add_argument('--loss', type=str, default='CE', choices=['CE','LS','CE+D'], help='type of loss function')
    parser.add_argument('--model', type=str, default='DeepLabV3+', choices=['D-LinkNet','GCN','DeepLabV3+','FCN','UNet','ENet'], help='model to train')
    parser.add_argument('--schedule_mode', type=str, default='poly', help='which scheduler to apply')
    parser.add_argument('--init_eval', type=str2bool, default=False, help='whether to start with evaluation')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate of training')
    parser.add_argument('--dataset', type=str, default='Potsdam', choices=['Potsdam','UDD5','UDD6','Custom','ArsUDD','uavid'], help='which dataset to train')

    parser.add_argument('--resized_width', type=int, default=1024, help='width after resize')
    parser.add_argument('--resized_height', type=int, default=1024, help='height after resize')
    parser.add_argument('--resize', action='store_true', help='whether to resize the input image')
    
    parser.add_argument('--save_dir', type=str, default='./', help='path of saving directory')

    args = parser.parse_args()
    print(args)
    argsDict = args.__dict__
    project_base_dir = os.path.abspath(os.path.join(args.data_path, "../"))
    args_save_dir = os.path.join(project_base_dir,"saved_args")
    if not os.path.exists(args_save_dir):
        os.makedirs(args_save_dir)
    with open(os.path.join(args_save_dir,"AerialSeg_args.txt"), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    my_trainer = Trainer(args)
    if args.mode=='train':
        my_trainer.run()
    else:
        my_trainer.validate(epoch=-1,save=True)

if __name__ == "__main__":
    main()