import argparse
arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


main_arg = add_argument_group('Main')


main_arg.add_argument('--mode', type=str,
                      default='Training',
                      help='Running mode')

main_arg.add_argument('--save_dir', type=str,
                      default='/projects/ovcare/classification/ywang/project_save/gan_eval_save/',
                      help='Save model weights')

main_arg.add_argument('--log_dir', type=str,
                      default='/projects/ovcare/classification/ywang/project_log/gan_eval_log/',
                      help='TensorBoard directory')

main_arg.add_argument('--dataset_dir', type=str,
                      default='/projects/ovcare/classification/ywang/gan_tcga_dataset/',
                      help='Slide and patch id files directory')

main_arg.add_argument('--deep_model', type=str,
                      default='DeepModel')

main_arg.add_argument('--deep_classifier', type=str,
                      default='vgg19_bn')


main_arg.add_argument('--train_ids_file_name', type=str,
                      default='patch_ids/baseline_0_train_ids.txt',
                      help='Training patch path ids')

main_arg.add_argument('--val_ids_file_name', type=str,
                      default='patch_ids/0_val_ids.txt',
                      help='Validation patch path ids')

main_arg.add_argument('--test_ids_file_name', type=str,
                      default='patch_ids/0_test_ids.txt',
                      help='Testing patch path ids')

main_arg.add_argument('--preload_image_file_name',
                      type=str, default='images.h5')

main_arg.add_argument('--aug_target_file_name',
                      type=str, default='progressive_resizing.hdf5')

main_arg.add_argument('--aug_real_file_name',
                      type=str, default='progressive_resizing.hdf5')

main_arg.add_argument('--aug_fake_file_name',
                      type=str, default='1024_fake.hdf5')

main_arg.add_argument('--model_name_prefix', type=str, default='')

main_arg.add_argument('--epoch', type=int,
                      default=5,
                      help='Number of epoches')

main_arg.add_argument('--batch_size', type=int,
                      default=32,
                      help='Batch size')

main_arg.add_argument('--eval_batch_size', type=int,
                      default=1,
                      help='Batch size')

main_arg.add_argument('--lr', type=float,
                      default=0.0002,
                      help='You know, the learning rate')

main_arg.add_argument('--l2_decay', type=float, default=0)

main_arg.add_argument('--rep_intv', type=int,
                      default=500,
                      help='Report interval')

main_arg.add_argument('--save_intv', type=int,
                      default=5,
                      help='Save model interval')

main_arg.add_argument('--patch_size', type=int,
                      default=256,
                      help='Patch size')

main_arg.add_argument('--n_eval_samples', type=int,
                      default=500,
                      help='Number of samples for eval during training')

main_arg.add_argument('--n_subtypes', type=int,
                      default=5)

main_arg.add_argument('--optim', type=str, default='Adam')

main_arg.add_argument('--continue_train', action='store_true')

main_arg.add_argument('--log_patches', action='store_true')

main_arg.add_argument('--use_pretrained', action='store_true')

main_arg.add_argument('--use_equalized_batch', action='store_true')

main_arg.add_argument('--test_augmentation', action='store_true')

main_arg.add_argument('--load_pretrained', action='store_true')

main_arg.add_argument('--load_model_id', type=str,
                      default='max_val_acc')

main_arg.add_argument('--is_tcga', action='store_true')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()
